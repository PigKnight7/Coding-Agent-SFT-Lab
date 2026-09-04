#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cc_agent.sft_validation import validate_paths  # noqa: E402

EXPECTED_MODEL_ID = "Qwen/Qwen3.5-2B"
EXPECTED_MODEL_TYPE = "qwen3_5"
EXPECTED_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
EXPECTED_TEMPLATE = "qwen3_5_nothink"
REQUIRED_VERSIONS = {
    "llamafactory": "0.9.5",
    "transformers": "5.5.0",
    "peft": "0.18.1",
}
CONFIG_PATHS = (
    PROJECT_ROOT / "configs" / "qwen3_5_2b_smoke.yaml",
    PROJECT_ROOT / "configs" / "qwen3_5_2b_lora_sft.yaml",
)
DATA_FILES = (
    PROJECT_ROOT / "data" / "llamafactory" / "train_alpaca.json",
    PROJECT_ROOT / "data" / "llamafactory" / "val_alpaca.json",
    PROJECT_ROOT / "data" / "llamafactory" / "test_alpaca.json",
    PROJECT_ROOT / "data" / "llamafactory" / "smoke_alpaca.json",
)
RAW_DATA_FILES = (
    PROJECT_ROOT / "data" / "sft" / "agent_traces_sft.jsonl",
    PROJECT_ROOT / "data" / "sft" / "swebench_lite_plan_sft.jsonl",
    PROJECT_ROOT / "data" / "sft" / "mbpp_strategy_sft.jsonl",
    PROJECT_ROOT / "data" / "sft" / "mbpp_sft.jsonl",
    PROJECT_ROOT / "data" / "sft" / "humaneval_sft.jsonl",
)
SFT_REQUIREMENTS_PATH = PROJECT_ROOT / "requirements-qwen3_5-sft.txt"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str

    def render(self) -> str:
        return f"[{self.status}] {self.name}: {self.detail}"


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return value


def validate_config_contract(config: dict[str, Any], *, smoke: bool) -> list[str]:
    expected = {
        "model_name_or_path": EXPECTED_MODEL_ID,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_target": "all",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "freeze_language_model": False,
        "template": EXPECTED_TEMPLATE,
        "enable_thinking": False,
        "train_on_prompt": False,
        "cutoff_len": 4096,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1.0e-4,
        "bf16": True,
        "gradient_checkpointing": True,
        "seed": 42,
        "data_seed": 42,
    }
    errors = [f"{key} must be {value!r}, got {config.get(key)!r}" for key, value in expected.items() if config.get(key) != value]
    if smoke:
        if config.get("dataset") != "coding_agent_smoke":
            errors.append("Smoke dataset must be coding_agent_smoke")
        if config.get("max_steps") != 2:
            errors.append("Smoke max_steps must be 2")
    else:
        if config.get("dataset") != "coding_agent_train" or config.get("eval_dataset") != "coding_agent_val":
            errors.append("Formal run must use coding_agent_train and coding_agent_val")
        if config.get("num_train_epochs") != 1.0:
            errors.append("Formal num_train_epochs must be 1.0")
        if "coding_agent_test" in {config.get("dataset"), config.get("eval_dataset")}:
            errors.append("coding_agent_test must not be used for training or validation")
    if config.get("adapter_name_or_path"):
        errors.append("Training must start from the base model, not a previous adapter")
    return errors


def group_overlap(data_files: Iterable[str | Path]) -> dict[str, list[str]]:
    groups: dict[str, set[str]] = {}
    for value in data_files:
        path = Path(value)
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON array")
        missing = sum(not isinstance(row, dict) or not row.get("group_id") for row in rows)
        if missing:
            raise ValueError(f"{path} contains {missing} rows without group_id")
        groups[path.stem] = {str(row["group_id"]) for row in rows}
    names = list(groups)
    return {
        f"{left}:{right}": sorted(groups[left] & groups[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }


def static_checks(config_path: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    python_status = "PASS" if sys.version_info >= (3, 11) else "FAIL"
    results.append(CheckResult("python", python_status, f">=3.11 required, found={sys.version.split()[0]}"))
    try:
        pins = {
            name.strip().lower(): version.strip()
            for raw_line in SFT_REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
            if raw_line.strip() and not raw_line.lstrip().startswith("#") and "==" in raw_line
            for name, version in (raw_line.split("==", 1),)
        }
        expected_pins = {name: version for name, version in REQUIRED_VERSIONS.items()}
        if pins != expected_pins:
            raise ValueError(f"expected={expected_pins}, found={pins}")
        results.append(CheckResult("dependency_pins", "PASS", json.dumps(pins, sort_keys=True)))
    except Exception as exc:
        results.append(CheckResult("dependency_pins", "FAIL", str(exc)))
    configs: list[dict[str, Any]] = []
    for path in CONFIG_PATHS:
        try:
            config = load_config(path)
            configs.append(config)
            errors = validate_config_contract(config, smoke=path.name.endswith("smoke.yaml"))
            if errors:
                results.append(CheckResult(f"config:{path.name}", "FAIL", "; ".join(errors)))
            else:
                results.append(CheckResult(f"config:{path.name}", "PASS", "training contract matches"))
        except Exception as exc:
            results.append(CheckResult(f"config:{path.name}", "FAIL", str(exc)))
    if len(configs) == 2:
        outputs = [str(config.get("output_dir", "")) for config in configs]
        status = "PASS" if outputs[0] and outputs[1] and outputs[0] != outputs[1] else "FAIL"
        results.append(CheckResult("independent_outputs", status, f"smoke={outputs[0]!r}, formal={outputs[1]!r}"))

    try:
        selected = load_config(config_path)
        results.append(CheckResult("selected_config", "PASS", str(config_path)))
        dataset_dir = _resolve_project_path(str(selected.get("dataset_dir", "")))
        info_path = dataset_dir / "dataset_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        required_datasets = {str(selected.get("dataset", ""))}
        if selected.get("eval_dataset"):
            required_datasets.add(str(selected["eval_dataset"]))
        missing = sorted(required_datasets - set(info))
        if missing:
            raise ValueError(f"dataset_info.json is missing {missing}")
        for name in required_datasets:
            data_path = dataset_dir / info[name]["file_name"]
            if not data_path.is_file() or data_path.stat().st_size == 0:
                raise ValueError(f"dataset {name} file is missing or empty: {data_path}")
        results.append(CheckResult("dataset_registry", "PASS", ", ".join(sorted(required_datasets))))
    except Exception as exc:
        results.append(CheckResult("dataset_registry", "FAIL", str(exc)))

    report = validate_paths((*RAW_DATA_FILES, *DATA_FILES), require_explicit_task_type=True)
    if report.errors:
        preview = "; ".join(issue.render() for issue in report.errors[:3])
        results.append(CheckResult("strict_data_validation", "FAIL", f"{len(report.errors)} errors: {preview}"))
    else:
        results.append(
            CheckResult(
                "strict_data_validation",
                "PASS",
                f"records={report.records_checked}, warnings={len(report.warnings)}",
            )
        )
    try:
        overlap = group_overlap(DATA_FILES[:3])
        leaking = {name: ids for name, ids in overlap.items() if ids}
        if leaking:
            results.append(CheckResult("task_leakage", "FAIL", json.dumps(leaking, ensure_ascii=False)))
        else:
            results.append(CheckResult("task_leakage", "PASS", "Train/Validation/Test group intersections are empty"))
    except Exception as exc:
        results.append(CheckResult("task_leakage", "FAIL", str(exc)))
    try:
        smoke_rows = json.loads(DATA_FILES[3].read_text(encoding="utf-8"))
        status = "PASS" if isinstance(smoke_rows, list) and len(smoke_rows) == 32 else "FAIL"
        results.append(CheckResult("smoke_size", status, f"records={len(smoke_rows) if isinstance(smoke_rows, list) else 'invalid'}"))
    except Exception as exc:
        results.append(CheckResult("smoke_size", "FAIL", str(exc)))
    return results


def runtime_checks(config_path: Path, model_path: Path, output_dir: Path, min_gpu_memory_gib: float) -> list[CheckResult]:
    results: list[CheckResult] = []
    if sys.version_info < (3, 11):
        results.append(CheckResult("python", "FAIL", f">=3.11 required, found {sys.version.split()[0]}"))
    else:
        results.append(CheckResult("python", "PASS", sys.version.split()[0]))
    for package, expected in REQUIRED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
            status = "PASS" if actual == expected else "FAIL"
            results.append(CheckResult(f"dependency:{package}", status, f"required={expected}, found={actual}"))
        except importlib.metadata.PackageNotFoundError:
            results.append(CheckResult(f"dependency:{package}", "FAIL", "not installed"))

    try:
        import torch

        torch_version = importlib.metadata.version("torch")
        torch_status = "PASS" if _numeric_version(torch_version) >= (2, 4, 0) else "FAIL"
        results.append(CheckResult("dependency:torch", torch_status, f">=2.4.0 required, found={torch_version}"))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        properties = torch.cuda.get_device_properties(0)
        memory_gib = properties.total_memory / (1024**3)
        if memory_gib < min_gpu_memory_gib:
            raise RuntimeError(f"GPU memory {memory_gib:.2f} GiB is below {min_gpu_memory_gib:.2f} GiB")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("selected GPU/PyTorch build does not support BF16")
        results.append(CheckResult("cuda_bf16", "PASS", f"{properties.name}, memory={memory_gib:.2f} GiB"))
    except Exception as exc:
        results.append(CheckResult("cuda_bf16", "FAIL", str(exc)))

    try:
        _validate_local_model(model_path)
        results.append(CheckResult("local_model", "PASS", str(model_path)))
    except Exception as exc:
        results.append(CheckResult("local_model", "FAIL", str(exc)))

    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"output directory must be absent or empty: {output_dir}")
        if not os.access(output_dir.parent, os.W_OK):
            raise ValueError(f"output parent is not writable: {output_dir.parent}")
        results.append(CheckResult("output_directory", "PASS", str(output_dir)))
    except Exception as exc:
        results.append(CheckResult("output_directory", "FAIL", str(exc)))

    config = load_config(config_path)
    config["model_name_or_path"] = str(model_path)
    config["output_dir"] = str(output_dir)
    parsed_args = None
    try:
        from llamafactory.hparams import get_train_args

        parsed_args = get_train_args(config)
        results.append(CheckResult("llamafactory_parse", "PASS", config_path.name))
    except Exception as exc:
        results.append(CheckResult("llamafactory_parse", "FAIL", str(exc)))

    try:
        from llamafactory.data.template import get_template_and_fix_tokenizer
        from llamafactory.hparams import DataArguments
        from transformers import AutoConfig, AutoProcessor

        model_config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        architecture = (getattr(model_config, "architectures", None) or [None])[0]
        if model_config.model_type != EXPECTED_MODEL_TYPE or architecture != EXPECTED_ARCHITECTURE:
            raise ValueError(f"unexpected model class: model_type={model_config.model_type!r}, architecture={architecture!r}")
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        tokenizer = getattr(processor, "tokenizer", processor)
        data_args = DataArguments(template=EXPECTED_TEMPLATE, train_on_prompt=False, enable_thinking=False)
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        sample = json.loads(DATA_FILES[3].read_text(encoding="utf-8"))[0]
        messages = [
            {"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"},
            {"role": "assistant", "content": sample["output"]},
        ]
        prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages, system=sample.get("system", ""))
        if not prompt_ids or not response_ids:
            raise ValueError("template produced an empty prompt or response")
        results.append(CheckResult("chat_template", "PASS", f"prompt_tokens={len(prompt_ids)}, response_tokens={len(response_ids)}"))
    except Exception as exc:
        results.append(CheckResult("chat_template", "FAIL", str(exc)))

    if parsed_args is not None:
        try:
            from llamafactory.model import load_model, load_tokenizer

            model_args, _data_args, _training_args, finetuning_args, _generating_args = parsed_args
            tokenizer_module = load_tokenizer(model_args)
            model = load_model(
                tokenizer_module["tokenizer"],
                model_args,
                finetuning_args,
                is_trainable=True,
            )
            summary = summarize_trainable_parameters(model)
            print(render_trainable_parameter_summary(summary))
            if summary["visual_trainable_names"]:
                preview = ", ".join(summary["visual_trainable_names"][:10])
                raise ValueError(f"visual/projector parameters are trainable: {preview}")
            if summary["trainable_parameters"] == 0:
                raise ValueError("LoRA initialization produced zero trainable parameters")
            results.append(
                CheckResult(
                    "trainable_parameters",
                    "PASS",
                    f"trainable={summary['trainable_parameters']:,}, total={summary['total_parameters']:,}, "
                    f"tensors={summary['trainable_tensors']}",
                )
            )
        except Exception as exc:
            results.append(CheckResult("trainable_parameters", "FAIL", str(exc)))
    return results


_VISUAL_PARAMETER_MARKERS = (
    ".visual.",
    "visual.",
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
)


def summarize_trainable_parameters(model: Any) -> dict[str, Any]:
    """Summarize the exact post-LLaMA-Factory trainable set without importing torch."""
    total = 0
    trainable = 0
    trainable_names: list[str] = []
    visual_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
            lowered = name.lower()
            if any(marker in lowered for marker in _VISUAL_PARAMETER_MARKERS):
                visual_names.append(name)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_tensors": len(trainable_names),
        "trainable_names": trainable_names,
        "visual_trainable_names": visual_names,
    }


def render_trainable_parameter_summary(summary: dict[str, Any]) -> str:
    ratio = 100.0 * summary["trainable_parameters"] / max(summary["total_parameters"], 1)
    names = "\n".join(f"  - {name}" for name in summary["trainable_names"])
    return (
        "trainable parameter audit:\n"
        f"  trainable={summary['trainable_parameters']:,}\n"
        f"  total={summary['total_parameters']:,}\n"
        f"  ratio={ratio:.6f}%\n"
        f"  tensors={summary['trainable_tensors']}\n"
        f"{names}"
    )


def _validate_local_model(path: Path) -> None:
    if not path.is_dir():
        raise ValueError(f"MODEL_PATH must be an existing local directory, got {path}")
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"local model is missing {missing}")
    has_weights = any(path.glob("*.safetensors")) or (path / "model.safetensors.index.json").is_file()
    if not has_weights:
        raise ValueError("local model has no safetensors weights")


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _numeric_version(value: str) -> tuple[int, int, int]:
    parts = value.split("+", 1)[0].split(".")
    numbers = []
    for part in parts[:3]:
        digits = "".join(character for character in part if character.isdigit())
        numbers.append(int(digits or 0))
    return tuple((numbers + [0, 0, 0])[:3])  # type: ignore[return-value]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight Qwen3.5-2B LLaMA-Factory SFT without starting training.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATHS[0])
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-gpu-memory-gib", type=float, default=20.0)
    parser.add_argument("--static-only", action="store_true", help="Run CPU-only config/data checks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = _resolve_project_path(str(args.config))
    results = static_checks(config_path)
    if not args.static_only:
        if args.model_path is None:
            results.append(CheckResult("local_model", "FAIL", "--model-path is required for runtime checks"))
        else:
            config = load_config(config_path)
            output = args.output_dir or _resolve_project_path(str(config["output_dir"]))
            model = args.model_path if args.model_path.is_absolute() else PROJECT_ROOT / args.model_path
            results.extend(runtime_checks(config_path, model, output, args.min_gpu_memory_gib))
    for result in results:
        print(result.render())
    failures = sum(result.status == "FAIL" for result in results)
    print(f"summary: checks={len(results)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
