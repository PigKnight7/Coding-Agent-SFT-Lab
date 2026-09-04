from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cc_agent.data_builder import recover_task_from_plan  # noqa: E402
from cc_agent.sft_validation import infer_task_type, validate_record  # noqa: E402


SYSTEM_PROMPT = """你是一个 Claude Code 风格的代码仓库 Agent。
你需要根据用户任务、仓库上下文和历史工具轨迹，输出严格可解析的 JSON。
优先选择最小、安全、可测试的修改方案。"""
DEFAULT_INPUTS = [
    "data/sft/agent_traces_sft.jsonl",
    "data/sft/swebench_lite_plan_sft.jsonl",
    "data/sft/mbpp_strategy_sft.jsonl",
    "data/sft/mbpp_sft.jsonl",
    "data/sft/humaneval_sft.jsonl",
]
OPTIONAL_PATCH_INPUT = "data/sft/swebench_lite_patch_sft.jsonl"
SPLIT_RATIOS = {"train": 0.90, "validation": 0.05, "test": 0.05}


class DataPreparationError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, group-split, and convert project SFT data to Alpaca format.")
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--include-patch", action="store_true", help="Optionally include SWE-bench patch data")
    parser.add_argument("--output-dir", default="data/llamafactory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-size", type=int, default=32)
    parser.add_argument("--rewrite-clean-sources", action="store_true")
    args = parser.parse_args()
    inputs = list(args.inputs)
    if args.include_patch and OPTIONAL_PATCH_INPUT not in inputs:
        inputs.append(OPTIONAL_PATCH_INPUT)
    result = prepare_datasets(
        inputs=inputs,
        output_dir=args.output_dir,
        seed=args.seed,
        smoke_size=args.smoke_size,
        rewrite_clean_sources=args.rewrite_clean_sources,
        include_patch=args.include_patch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def prepare_datasets(
    *,
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    seed: int = 42,
    smoke_size: int = 32,
    rewrite_clean_sources: bool = False,
    include_patch: bool = False,
) -> dict[str, Any]:
    input_paths = [Path(value) for value in inputs]
    task_ids = _load_task_id_map(PROJECT_ROOT)
    source_stats: dict[str, dict[str, int]] = {}
    repair_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    prepared: list[tuple[Path, dict[str, Any]]] = []

    for path in input_paths:
        rows = _read_jsonl_strict(path)
        kept = 0
        for row in rows:
            cleaned, repairs, exclusion = clean_record(row, path, task_ids=task_ids)
            repair_counts.update(repairs)
            if exclusion:
                excluded_counts[exclusion] += 1
                continue
            assert cleaned is not None
            prepared.append((path, cleaned))
            kept += 1
        source_stats[path.name] = {"before": len(rows), "after_cleaning": kept}

    deduplicated, duplicate_counts = _deduplicate(prepared)
    for path, count in duplicate_counts.items():
        if count:
            source_stats[path.name]["duplicates_removed"] = count
    for _, row in deduplicated:
        row["group_id"] = derive_group_id(row)
    _assert_clean_records(deduplicated)

    if rewrite_clean_sources:
        by_source: dict[Path, list[dict[str, Any]]] = defaultdict(list)
        for path, row in deduplicated:
            by_source[path].append(row)
        for path in input_paths:
            _write_jsonl(path, by_source.get(path, []))

    records = [row for _, row in deduplicated]
    splits = split_by_group(records, seed=seed)
    assert_no_group_overlap(splits)
    smoke = select_smoke_records(splits["train"], size=smoke_size, seed=seed)
    converted = {name: [convert_record(row) for row in rows] for name, rows in splits.items()}
    converted_smoke = [convert_record(row) for row in smoke]
    _assert_clean_alpaca({**converted, "smoke": converted_smoke})

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": destination / "train_alpaca.json",
        "validation": destination / "val_alpaca.json",
        "test": destination / "test_alpaca.json",
        "smoke": destination / "smoke_alpaca.json",
    }
    for name, path in output_paths.items():
        payload = converted_smoke if name == "smoke" else converted[name]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "dataset_info.json").write_text(
        json.dumps(_dataset_info(output_paths), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_before = sum(value["before"] for value in source_stats.values())
    total_after_cleaning = sum(value["after_cleaning"] for value in source_stats.values())
    stats: dict[str, Any] = {
        "seed": seed,
        "split_ratios": SPLIT_RATIOS,
        "include_patch": include_patch,
        "inputs": source_stats,
        "cleaning": {
            "before": total_before,
            "after_cleaning": total_after_cleaning,
            "excluded": sum(excluded_counts.values()),
            "excluded_by_reason": dict(sorted(excluded_counts.items())),
            "repairs": dict(sorted(repair_counts.items())),
            "exact_duplicates_removed": sum(duplicate_counts.values()),
            "final": len(records),
        },
        "splits": {
            name: {
                "records": len(rows),
                "groups": len({row["group_id"] for row in rows}),
                "task_types": dict(sorted(Counter(row["task_type"] for row in rows).items())),
            }
            for name, rows in splits.items()
        },
        "smoke": {
            "records": len(smoke),
            "groups": len({row["group_id"] for row in smoke}),
            "task_types": dict(sorted(Counter(row["task_type"] for row in smoke).items())),
        },
        "group_overlap": _group_overlaps(splits),
    }
    (destination / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return stats


def clean_record(
    original: dict[str, Any],
    source_path: Path,
    *,
    task_ids: dict[str, tuple[str, str]],
) -> tuple[dict[str, Any] | None, Counter[str], str | None]:
    record = json.loads(json.dumps(original, ensure_ascii=False))
    repairs: Counter[str] = Counter()
    task_type = infer_task_type(record)
    if task_type is None:
        return None, repairs, "unknown_task_type"
    if record.get("task_type") != task_type:
        record["task_type"] = task_type
        repairs["task_type_added"] += 1

    output = record.get("output")
    if task_type == "tool_strategy" and isinstance(output, dict):
        for step in output.get("strategy", []):
            if isinstance(step, dict) and step.get("tool") == "replace_in_file 或 write_file":
                step["tool"] = "replace_in_file"
                repairs["strategy_tool_normalized"] += 1

    if source_path.name == "humaneval_sft.jsonl" and isinstance(output, dict):
        arguments = output.get("arguments")
        if isinstance(arguments, dict) and arguments.get("path") == "solution.py":
            content = arguments.get("content")
            if isinstance(content, str):
                repaired = repair_humaneval_code(content)
                if repaired != content:
                    arguments["content"] = repaired
                    repairs["humaneval_newline_repaired"] += 1

    input_value = record.get("input")
    if isinstance(input_value, dict) and "unknown task" in str(input_value.get("task", "")).lower():
        recovered = _recover_trace_task(record)
        if not recovered:
            return None, repairs, "unknown_task_unrecoverable"
        input_value["task"] = recovered
        repairs["unknown_task_recovered"] += 1

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        record["metadata"] = metadata
    metadata["source_file"] = source_path.name
    task = _task_text(record)
    identity = task_ids.get(_normalize_task(task)) if task else None
    if identity:
        metadata.setdefault("benchmark_source", identity[0])
        metadata.setdefault("benchmark_id", identity[1])
        if source_path.name in {"mbpp_sft.jsonl", "humaneval_sft.jsonl"}:
            metadata.setdefault("source", identity[0])
            metadata.setdefault("id", identity[1])
    return record, repairs, None


def repair_humaneval_code(content: str) -> str:
    if _valid_python(content):
        return content
    boundaries = [match.end() for match in re.finditer(r'(?:"""|\'\'\')(?= {4}\S)', content)]
    for boundary in boundaries:
        candidate = content[:boundary] + "\n" + content[boundary:]
        if _valid_python(candidate):
            return candidate
    return content


def derive_group_id(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    if metadata.get("benchmark_source") and metadata.get("benchmark_id"):
        return f"benchmark:{metadata['benchmark_source']}:{metadata['benchmark_id']}"
    if record.get("task_type") in {"swebench_plan", "swebench_patch"} and metadata.get("id"):
        return f"issue:{metadata['id']}"
    task = _task_text(record)
    if task:
        return "task:" + hashlib.sha256(_normalize_task(task).encode("utf-8")).hexdigest()[:20]
    if metadata.get("trace_file"):
        return "trace:" + hashlib.sha256(str(metadata["trace_file"]).encode("utf-8")).hexdigest()[:20]
    payload = json.dumps(_training_payload(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "record:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def split_by_group(records: list[dict[str, Any]], *, seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(record)
    group_ids = sorted(grouped)
    random.Random(seed).shuffle(group_ids)
    train_target = round(len(records) * SPLIT_RATIOS["train"])
    validation_target = round(len(records) * SPLIT_RATIOS["validation"])
    train_groups, remaining = _select_groups_near_target(group_ids, grouped, train_target)
    validation_groups, test_groups = _select_groups_near_target(remaining, grouped, validation_target)
    assignments = {"train": train_groups, "validation": validation_groups, "test": test_groups}
    result = {
        name: [row for group_id in selected for row in grouped[group_id]]
        for name, selected in assignments.items()
    }
    return result


def _select_groups_near_target(
    group_ids: list[str],
    grouped: dict[str, list[dict[str, Any]]],
    target: int,
) -> tuple[list[str], list[str]]:
    """Select whole groups with a deterministic record count nearest to target."""
    reachable: dict[int, tuple[int, str] | None] = {0: None}
    for group_id in group_ids:
        size = len(grouped[group_id])
        for current in sorted(tuple(reachable), reverse=True):
            candidate = current + size
            if candidate <= target and candidate not in reachable:
                reachable[candidate] = (current, group_id)
        if target in reachable:
            break
    selected_ids: set[str] = set()
    current = max(reachable)
    while current:
        previous, group_id = reachable[current]  # type: ignore[misc]
        selected_ids.add(group_id)
        current = previous
    selected = [group_id for group_id in group_ids if group_id in selected_ids]
    remaining = [group_id for group_id in group_ids if group_id not in selected_ids]
    return selected, remaining


def assert_no_group_overlap(splits: dict[str, list[dict[str, Any]]]) -> None:
    overlaps = _group_overlaps(splits)
    if any(overlaps.values()):
        raise DataPreparationError(f"Task groups overlap across splits: {overlaps}")


def select_smoke_records(records: list[dict[str, Any]], *, size: int = 32, seed: int = 42) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record["task_type"]].append(record)
    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    task_types = sorted(buckets)
    target = min(size, len(records))
    while len(selected) < target and task_types:
        for task_type in list(task_types):
            if buckets[task_type]:
                selected.append(buckets[task_type].pop())
                if len(selected) == target:
                    break
            else:
                task_types.remove(task_type)
    return selected


def convert_record(record: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    source_name = source or str(record.get("metadata", {}).get("source_file", "unknown"))
    group_id = record.get("group_id") or derive_group_id(record)
    converted: dict[str, Any] = {
        "system": SYSTEM_PROMPT,
        "instruction": f"{record['instruction']}\n\n来源文件: {source_name}",
        "input": json.dumps(record["input"], ensure_ascii=False, indent=2),
        "output": json.dumps(record["output"], ensure_ascii=False, indent=2),
        "task_type": record["task_type"],
        "group_id": group_id,
    }
    if isinstance(record.get("metadata"), dict):
        converted["metadata"] = record["metadata"]
    return converted


def _read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DataPreparationError(f"Missing input file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataPreparationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise DataPreparationError(f"{path}:{line_number}: sample must be an object")
        rows.append(row)
    return rows


def _load_task_id_map(project_root: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for name in ("mbpp_tasks.jsonl", "humaneval_tasks.jsonl"):
        path = project_root / "data" / "tasks" / name
        if path.exists():
            for row in _read_jsonl_strict(path):
                task = row.get("task")
                if isinstance(task, str) and task.strip():
                    result[_normalize_task(task)] = (str(row.get("source", "unknown")), str(row.get("id", "unknown")))
    return result


def _recover_trace_task(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    trace_file = metadata.get("trace_file")
    if isinstance(trace_file, str):
        path = PROJECT_ROOT / trace_file
        if path.exists():
            for row in _read_jsonl_strict(path):
                if row.get("event") == "repo_indexed":
                    task = row.get("payload", {}).get("task")
                    if isinstance(task, str) and task.strip():
                        return task.strip()
    input_value = record.get("input")
    plan = input_value.get("plan", "") if isinstance(input_value, dict) else ""
    return recover_task_from_plan(str(plan))


def _deduplicate(rows: list[tuple[Path, dict[str, Any]]]) -> tuple[list[tuple[Path, dict[str, Any]]], Counter[Path]]:
    seen: set[str] = set()
    result: list[tuple[Path, dict[str, Any]]] = []
    duplicates: Counter[Path] = Counter()
    for path, row in rows:
        key = json.dumps(_training_payload(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key in seen:
            duplicates[path] += 1
        else:
            seen.add(key)
            result.append((path, row))
    return result, duplicates


def _training_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in ("task_type", "instruction", "input", "output")}


def _task_text(record: dict[str, Any]) -> str:
    input_value = record.get("input")
    if isinstance(input_value, dict):
        for key in ("task", "problem_statement"):
            value = input_value.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _normalize_task(task: str) -> str:
    return " ".join(task.lower().split())


def _valid_python(content: str) -> bool:
    try:
        ast.parse(content)
        return True
    except (SyntaxError, ValueError):
        return False


def _assert_clean_records(rows: list[tuple[Path, dict[str, Any]]]) -> None:
    errors = [
        issue
        for path, row in rows
        for issue in validate_record(row, filename=path, require_explicit_task_type=True)
        if issue.severity == "error"
    ]
    if errors:
        preview = "\n".join(issue.render() for issue in errors[:20])
        raise DataPreparationError(f"Cleaned source validation failed with {len(errors)} error(s):\n{preview}")


def _assert_clean_alpaca(splits: dict[str, list[dict[str, Any]]]) -> None:
    errors = [
        issue
        for name, rows in splits.items()
        for index, row in enumerate(rows, 1)
        for issue in validate_record(row, filename=f"{name}_alpaca.json", line=index, require_explicit_task_type=True)
        if issue.severity == "error"
    ]
    if errors:
        preview = "\n".join(issue.render() for issue in errors[:20])
        raise DataPreparationError(f"Alpaca validation failed with {len(errors)} error(s):\n{preview}")


def _group_overlaps(splits: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    groups = {name: {row["group_id"] for row in rows} for name, rows in splits.items()}
    names = list(groups)
    return {
        f"{left}_{right}": sorted(groups[left] & groups[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }


def _dataset_info(paths: dict[str, Path]) -> dict[str, Any]:
    dataset_names = {
        "train": "coding_agent_train",
        "validation": "coding_agent_val",
        "test": "coding_agent_test",
        "smoke": "coding_agent_smoke",
    }
    return {
        dataset_names[name]: {
            "file_name": path.name,
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"},
        }
        for name, path in paths.items()
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
