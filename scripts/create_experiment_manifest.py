#!/usr/bin/env python3
"""Write a reproducibility manifest immediately before a training run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ("llamafactory", "transformers", "peft", "torch")
DATA_ARTIFACTS = (
    PROJECT_ROOT / "data" / "llamafactory" / "train_alpaca.json",
    PROJECT_ROOT / "data" / "llamafactory" / "val_alpaca.json",
    PROJECT_ROOT / "data" / "llamafactory" / "test_alpaca.json",
    PROJECT_ROOT / "data" / "llamafactory" / "smoke_alpaca.json",
    PROJECT_ROOT / "data" / "llamafactory" / "dataset_info.json",
    PROJECT_ROOT / "data" / "llamafactory" / "dataset_stats.json",
)
CONFIG_ARTIFACTS = (
    PROJECT_ROOT / "configs" / "qwen3_5_2b_smoke.yaml",
    PROJECT_ROOT / "configs" / "qwen3_5_2b_lora_sft.yaml",
)
TRAINING_KEYS = (
    "model_name_or_path",
    "stage",
    "finetuning_type",
    "lora_rank",
    "lora_alpha",
    "lora_target",
    "freeze_vision_tower",
    "freeze_multi_modal_projector",
    "freeze_language_model",
    "dataset",
    "eval_dataset",
    "template",
    "enable_thinking",
    "train_on_prompt",
    "cutoff_len",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "max_steps",
    "num_train_epochs",
    "bf16",
    "gradient_checkpointing",
    "seed",
    "data_seed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.rstrip()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _accelerator_info() -> dict[str, Any]:
    try:
        import torch

        devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                    }
                )
        return {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "devices": devices,
        }
    except Exception as exc:
        return {"cuda_available": False, "error": f"{type(exc).__name__}: {exc}", "devices": []}


def build_manifest(config_path: Path, model_path: Path, output_dir: Path, run_kind: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    missing = [str(path) for path in (*DATA_ARTIFACTS, *CONFIG_ARTIFACTS) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing data artifact(s): {missing}")
    diff = subprocess.run(
        ("git", "diff", "--binary", "HEAD"),
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_kind": run_kind,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "status_porcelain": _git("status", "--porcelain=v1", "--untracked-files=all").splitlines(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        },
        "artifacts": {
            "selected_config": str(config_path),
            "configs": {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in CONFIG_ARTIFACTS},
            "datasets": {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in DATA_ARTIFACTS},
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
            "accelerator": _accelerator_info(),
        },
        "model_path": str(model_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "training": {key: config[key] for key in TRAINING_KEYS if key in config},
    }


def write_manifest(config_path: Path, model_path: Path, output_dir: Path, run_kind: str) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config_path, model_path, output_dir, run_kind)
    destination = output_dir / "experiment_manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    destination = write_manifest(args.config, args.model_path, args.output_dir, args.run_kind)
    print(f"experiment_manifest={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
