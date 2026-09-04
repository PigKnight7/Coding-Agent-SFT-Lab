#!/usr/bin/env python3
"""Reproducible Base-vs-SFT evaluation on the held-out coding_agent_test split."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cc_agent.sft_validation import registered_tool_names, validate_record  # noqa: E402
from scripts.verify_sft_adapter import latest_adapter_path  # noqa: E402

TEST_DATASET_NAME = "coding_agent_test"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "llamafactory" / "test_alpaca.json"
DATASET_INFO_PATH = PROJECT_ROOT / "data" / "llamafactory" / "dataset_info.json"
SUPPORTED_TASK_TYPES = ("tool_call", "tool_strategy", "swebench_plan")
SMOKE_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "qwen3_5_2b_smoke_lora"


@dataclass(frozen=True)
class GenerationSettings:
    template: str = "qwen3_5_nothink"
    enable_thinking: bool = False
    seed: int = 42
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: None = None
    top_p: None = None


def load_test_samples(path: Path = TEST_DATA_PATH) -> list[dict[str, Any]]:
    if path.resolve() != TEST_DATA_PATH.resolve():
        raise ValueError(f"Evaluation is locked to {TEST_DATASET_NAME}: {TEST_DATA_PATH}")
    registry = json.loads(DATASET_INFO_PATH.read_text(encoding="utf-8"))
    entry = registry.get(TEST_DATASET_NAME)
    if not isinstance(entry, dict) or entry.get("file_name") != TEST_DATA_PATH.name:
        raise ValueError(f"dataset_info.json does not map {TEST_DATASET_NAME} to {TEST_DATA_PATH.name}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("coding_agent_test must be a non-empty JSON array")
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"test row {index} is not an object")
        if row.get("task_type") not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"test row {index} has unsupported task_type={row.get('task_type')!r}")
        if not row.get("group_id"):
            raise ValueError(f"test row {index} has no group_id")
    return rows


def ensure_formal_adapter(adapter_dir: Path) -> Path:
    resolved = adapter_dir.resolve()
    if resolved == SMOKE_OUTPUT_PATH.resolve() or "smoke" in adapter_dir.name.lower():
        raise ValueError("The Smoke adapter must not be used as the formal evaluation result")
    return latest_adapter_path(adapter_dir)


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _protocol_issues(sample: dict[str, Any], prediction: dict[str, Any] | None) -> list[str]:
    if prediction is None:
        return ["invalid_json"]
    candidate = {
        "task_type": sample["task_type"],
        "instruction": sample["instruction"],
        "input": sample["input"],
        "output": prediction,
    }
    return sorted(
        {
            issue.error_type
            for issue in validate_record(candidate, require_explicit_task_type=True)
            if issue.severity == "error"
        }
    )


def _tool_sequence(value: dict[str, Any] | None) -> list[str] | None:
    if value is None or not isinstance(value.get("strategy"), list) or not value["strategy"]:
        return None
    tools: list[str] = []
    for step in value["strategy"]:
        if not isinstance(step, dict) or not isinstance(step.get("tool"), str):
            return None
        tools.append(step["tool"])
    return tools


def _target_files(reference: dict[str, Any] | None) -> list[str]:
    if reference is None or not isinstance(reference.get("plan"), str):
        return []
    return sorted(set(re.findall(r"(?<![\w.-])[\w./-]+\.py", reference["plan"])))


def score_response(sample: dict[str, Any], response: str, *, generation_error: str | None = None) -> dict[str, Any]:
    task_type = sample["task_type"]
    reference = parse_json_object(sample["output"])
    prediction = parse_json_object(response)
    success = generation_error is None and bool(response.strip())
    issues = _protocol_issues(sample, prediction) if success else ["generation_error" if generation_error else "empty_generation"]
    metrics: dict[str, float | None] = {
        "generation_success_rate": float(success),
        "json_valid_rate": float(prediction is not None),
        "protocol_valid_rate": float(success and not issues),
    }
    errors = list(issues)

    if task_type == "tool_call":
        predicted_tool = prediction.get("tool") if prediction else None
        reference_tool = reference.get("tool") if reference else None
        metrics["tool_name_accuracy"] = float(
            isinstance(predicted_tool, str)
            and isinstance(reference_tool, str)
            and predicted_tool == reference_tool
        )
        arguments_valid = prediction is not None and not any(
            issue in {"invalid_tool_arguments", "invalid_tool"} for issue in issues
        )
        metrics["tool_arguments_valid_rate"] = float(arguments_valid)
        if metrics["tool_name_accuracy"] == 0:
            errors.append("tool_name_mismatch")
        if not arguments_valid:
            errors.append("tool_arguments_invalid")

    elif task_type == "tool_strategy":
        predicted_sequence = _tool_sequence(prediction)
        reference_sequence = _tool_sequence(reference)
        sequence_valid = (
            predicted_sequence is not None
            and all(tool in registered_tool_names() for tool in predicted_sequence)
            and not any(issue in {"invalid_strategy", "invalid_tool"} for issue in issues)
        )
        metrics["tool_sequence_valid_rate"] = float(sequence_valid)
        metrics["tool_sequence_match_rate"] = float(
            predicted_sequence is not None
            and reference_sequence is not None
            and predicted_sequence == reference_sequence
        )
        if not sequence_valid:
            errors.append("tool_sequence_invalid")
        if metrics["tool_sequence_match_rate"] == 0:
            errors.append("tool_sequence_mismatch")

    elif task_type == "swebench_plan":
        plan_valid = (
            prediction is not None
            and isinstance(prediction.get("plan"), str)
            and bool(prediction["plan"].strip())
            and isinstance(prediction.get("validation"), str)
            and bool(prediction["validation"].strip())
            and "invalid_plan" not in issues
        )
        metrics["plan_structure_valid_rate"] = float(plan_valid)
        targets = _target_files(reference)
        metrics["target_file_hit_rate"] = (
            sum(target in response for target in targets) / len(targets) if targets else None
        )
        if not plan_valid:
            errors.append("plan_structure_invalid")
        if metrics["target_file_hit_rate"] == 0:
            errors.append("target_file_miss")

    return {
        "response": response,
        "generation_error": generation_error,
        "metrics": metrics,
        "errors": sorted(set(errors)),
    }


def aggregate_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    metric_values: dict[str, list[float]] = {}
    error_counts: Counter[str] = Counter()
    for row in rows:
        for name, value in row["metrics"].items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                metric_values.setdefault(name, []).append(float(value))
        error_counts.update(row["errors"])
    return {
        "samples": len(rows),
        "metrics": {name: sum(values) / len(values) for name, values in sorted(metric_values.items())},
        "error_counts": dict(sorted(error_counts.items())),
    }


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"base": {}, "sft": {}, "delta_sft_minus_base": {}}
    for model_kind in ("base", "sft"):
        summary[model_kind]["overall"] = aggregate_results(pair[model_kind] for pair in pairs)
        summary[model_kind]["by_task"] = {
            task: aggregate_results(pair[model_kind] for pair in pairs if pair["task_type"] == task)
            for task in SUPPORTED_TASK_TYPES
        }
    summary["delta_sft_minus_base"]["overall"] = _metric_delta(
        summary["base"]["overall"], summary["sft"]["overall"]
    )
    summary["delta_sft_minus_base"]["by_task"] = {
        task: _metric_delta(summary["base"]["by_task"][task], summary["sft"]["by_task"][task])
        for task in SUPPORTED_TASK_TYPES
    }
    summary["failures"] = [
        {"sample_id": pair["sample_id"], "task_type": pair["task_type"], model: pair[model]["errors"]}
        for pair in pairs
        for model in ("base", "sft")
        if pair[model]["errors"]
    ]
    return summary


def _metric_delta(base: dict[str, Any], sft: dict[str, Any]) -> dict[str, float]:
    common = set(base["metrics"]) & set(sft["metrics"])
    return {name: sft["metrics"][name] - base["metrics"][name] for name in sorted(common)}


def generation_contract(settings: GenerationSettings) -> dict[str, Any]:
    params = asdict(settings)
    return {"base": dict(params), "sft": dict(params)}


def assert_identical_generation_contract(contract: dict[str, Any]) -> None:
    if contract.get("base") != contract.get("sft"):
        raise ValueError("Base and SFT generation settings differ")
    if contract["base"].get("enable_thinking") is not False:
        raise ValueError("enable_thinking must be false")


def save_results(output_dir: Path, pairs: list[dict[str, Any]], summary: dict[str, Any], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sample_results.jsonl").open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    payload = {"metadata": metadata, **summary}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5-2B Base / SFT evaluation",
        "",
        f"- Dataset: `{payload['metadata']['dataset_name']}` ({payload['metadata']['sample_count']} samples)",
        f"- Seed: `{payload['metadata']['generation']['base']['seed']}`",
        "- Thinking: disabled for both models",
        "",
        "## Overall metrics",
        "",
        "| Metric | Base | SFT | Delta |",
        "|---|---:|---:|---:|",
    ]
    base = payload["base"]["overall"]["metrics"]
    sft = payload["sft"]["overall"]["metrics"]
    delta = payload["delta_sft_minus_base"]["overall"]
    for metric in sorted(set(base) | set(sft)):
        lines.append(f"| {metric} | {_fmt(base.get(metric))} | {_fmt(sft.get(metric))} | {_fmt(delta.get(metric))} |")
    lines.extend(["", "## Metrics by task", ""])
    for task in SUPPORTED_TASK_TYPES:
        lines.extend([f"### {task}", "", "| Metric | Base | SFT | Delta |", "|---|---:|---:|---:|"])
        task_base = payload["base"]["by_task"][task]["metrics"]
        task_sft = payload["sft"]["by_task"][task]["metrics"]
        task_delta = payload["delta_sft_minus_base"]["by_task"][task]
        for metric in sorted(set(task_base) | set(task_sft)):
            lines.append(
                f"| {metric} | {_fmt(task_base.get(metric))} | {_fmt(task_sft.get(metric))} | {_fmt(task_delta.get(metric))} |"
            )
        lines.append("")
    lines.extend(["## Failures", ""])
    if payload["failures"]:
        for failure in payload["failures"]:
            model = "base" if "base" in failure else "sft"
            lines.append(f"- `{failure['sample_id']}` / `{model}`: {', '.join(failure[model])}")
    else:
        lines.append("No failures recorded.")
    return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _input_device(model: Any) -> Any:
    return next(model.parameters()).device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_for_model(processor: Any, model: Any, samples: list[dict[str, Any]], settings: GenerationSettings) -> list[dict[str, Any]]:
    import torch

    outputs: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        _set_seed(settings.seed + index)
        messages = [
            {"role": "system", "content": sample.get("system", "")},
            {"role": "user", "content": f"{sample['instruction']}\n\n{sample.get('input', '')}"},
        ]
        started = time.perf_counter()
        try:
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=settings.enable_thinking,
            )
            inputs = processor(text=[prompt], return_tensors="pt").to(_input_device(model))
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=settings.max_new_tokens,
                    do_sample=settings.do_sample,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            if any(not torch.isfinite(score).all().item() for score in generated.scores):
                raise FloatingPointError("generation scores contain NaN or Inf")
            new_tokens = generated.sequences[0, inputs["input_ids"].shape[1] :]
            response = processor.decode(new_tokens, skip_special_tokens=True).strip()
            scored = score_response(sample, response)
            scored.update({"generated_tokens": int(new_tokens.numel()), "elapsed_seconds": time.perf_counter() - started})
        except Exception as exc:
            scored = score_response(sample, "", generation_error=f"{type(exc).__name__}: {exc}")
            scored.update({"generated_tokens": 0, "elapsed_seconds": time.perf_counter() - started})
        outputs.append(scored)
    return outputs


def evaluate(model_path: Path, adapter_dir: Path, output_dir: Path, settings: GenerationSettings) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    samples = load_test_samples()
    adapter_path = ensure_formal_adapter(adapter_dir)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }
    base_model = AutoModelForMultimodalLM.from_pretrained(model_path, **load_kwargs)
    base_model.eval()
    base_results = generate_for_model(processor, base_model, samples, settings)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    sft_base = AutoModelForMultimodalLM.from_pretrained(model_path, **load_kwargs)
    sft_model = PeftModel.from_pretrained(sft_base, adapter_path, is_trainable=False)
    sft_model.eval()
    sft_results = generate_for_model(processor, sft_model, samples, settings)

    pairs = [
        {
            "sample_id": str(sample["group_id"]),
            "sample_index": index,
            "task_type": sample["task_type"],
            "reference": sample["output"],
            "base": base_results[index],
            "sft": sft_results[index],
        }
        for index, sample in enumerate(samples)
    ]
    contract = generation_contract(settings)
    assert_identical_generation_contract(contract)
    metadata = {
        "dataset_name": TEST_DATASET_NAME,
        "dataset_path": str(TEST_DATA_PATH.relative_to(PROJECT_ROOT)),
        "dataset_sha256": hashlib.sha256(TEST_DATA_PATH.read_bytes()).hexdigest(),
        "sample_count": len(samples),
        "model_path": str(model_path),
        "adapter_path": str(adapter_path),
        "generation": contract,
    }
    save_results(output_dir, pairs, summarize_pairs(pairs), metadata)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Base and formal SFT adapter on coding_agent_test only.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = GenerationSettings(seed=args.seed, max_new_tokens=args.max_new_tokens)
    evaluate(args.model_path, args.adapter_dir, args.output_dir, settings)
    print(f"sample_results={args.output_dir / 'sample_results.jsonl'}")
    print(f"summary_json={args.output_dir / 'summary.json'}")
    print(f"summary_markdown={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
