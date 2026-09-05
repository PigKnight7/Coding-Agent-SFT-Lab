#!/usr/bin/env python3
"""Agentic RL precheck, training, resumption and held-out evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cc_agent.rl.artifacts import adapter_check, manifest, prepare_run
from cc_agent.rl.config import load_config
from cc_agent.rl.data import load_tasks, select


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("precheck", "train", "eval"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/agentic_rl_train.yaml")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--sft-adapter", type=Path)
    parser.add_argument("--rl-adapter", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-run", type=Path, help="Verified GPU Smoke run required before formal RL")
    parser.add_argument("--split", choices=("validation", "test"))
    args = parser.parse_args(argv)
    if args.static_only and args.mode != "precheck":
        parser.error("--static-only is only valid for precheck")
    if not args.static_only and (args.model_path is None or args.sft_adapter is None):
        parser.error("--model-path and --sft-adapter are required")
    if args.mode in ("train", "eval") and args.output_dir is None:
        parser.error("--output-dir is required")
    if args.mode == "eval" and args.rl_adapter is None:
        parser.error("--rl-adapter is required")
    if (args.resume or args.smoke_run) and args.mode != "train":
        parser.error("--resume is only valid for train")
    if args.mode != "eval" and (args.split is not None or args.rl_adapter is not None):
        parser.error("--split and --rl-adapter are evaluation-only")
    if args.mode == "eval" and args.split is None:
        args.split = "validation"
    return args


def runtime_check(model_path, adapter):
    import torch
    from cc_agent.rl.training import check_trl_api
    check_trl_api()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("First version supports one GPU/process only")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA and BF16 are required")
    model_config = json.loads((model_path / "config.json").read_text())
    text_config = model_config.get("text_config", {})
    if (model_config.get("model_type") != "qwen3_5" or
        model_config.get("architectures") != ["Qwen3_5ForConditionalGeneration"] or
        any(text_config.get(k) != v for k, v in {"hidden_size": 2048, "num_hidden_layers": 24, "intermediate_size": 6144}.items())):
        raise ValueError("Expected local Qwen3.5-2B model")
    index = model_path / "model.safetensors.index.json"
    weights = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    if any(not (model_path / p).is_file() for p in weights):
        raise ValueError("Local model weights incomplete; downloads are disabled")
    adapter_check(adapter, formal=True)
    # Fail before model loading if the host disallows namespaces or the sandbox Python lacks pytest.
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_rl_sandbox import check_sandbox
    check_sandbox()
    print("[PASS] TRL API, CUDA/BF16, local model, formal SFT adapter, isolated verifier controls")


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    selected_split = args.split if args.mode == "eval" else "train" if args.mode == "train" or not args.static_only else None
    tasks, excluded = load_tasks(ROOT, selected_split)
    print(json.dumps({"executable_tasks": dict(Counter(t.split for t in tasks)), "excluded": len(excluded)}, ensure_ascii=False))
    if args.static_only:
        print("[PASS] CPU configuration and data isolation. GPU/TRL runtime and OS sandbox execution NOT tested.")
        return 0
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true"})
    runtime_check(args.model_path, args.sft_adapter)
    if args.mode == "precheck":
        return 0
    output = args.output_dir.resolve()
    allowed = ROOT / ("outputs" if args.mode == "train" else "eval_results")
    if allowed not in output.parents:
        raise ValueError(f"Output must be a new run directory below {allowed.name}/")
    if args.mode == "train":
        training_tasks = select(tasks, "train")
        record = manifest(ROOT, training_tasks, config, args.model_path, args.sft_adapter, "train")
        if config.max_steps > 2 and not args.resume:
            if args.smoke_run is None:
                raise ValueError("Formal RL requires --smoke-run with passing two-step GPU evidence")
            from verify_rl_run import verify_run
            verify_run(args.smoke_run, 2)
            smoke = json.loads((args.smoke_run / "rl_manifest.json").read_text())
            for key in ("files", "dependencies", "tasks", "model_config_sha256", "model_files", "sft_adapter_sha256", "sft_adapter_config_sha256"):
                if smoke[key] != record[key]:
                    raise ValueError(f"Smoke/formal evidence mismatch: {key}")
            for key, value in record["config"].items():
                if key not in {"max_steps", "save_steps", "num_generations"} and smoke["config"][key] != value:
                    raise ValueError(f"Smoke/formal config mismatch: {key}")
        prepare_run(output, record, args.resume)
        from cc_agent.rl.training import train
        train(training_tasks, config, args.model_path, args.sft_adapter, output, args.resume)
    else:
        from cc_agent.rl.evaluation import evaluate
        from cc_agent.rl.artifacts import digest
        rl_adapter = adapter_check(args.rl_adapter)
        selected = select(tasks, args.split)
        record = manifest(ROOT, selected, config, args.model_path, args.sft_adapter, "eval")
        record["rl_adapter_sha256"] = digest(rl_adapter / "adapter_model.safetensors")
        prepare_run(output, record)
        evaluate(selected, config, args.model_path, {"sft": args.sft_adapter, "sft_dapo_grpo": rl_adapter}, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
