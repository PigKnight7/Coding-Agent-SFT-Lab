#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cc_agent.sft_validation import ValidationReport, validate_paths  # noqa: E402


DEFAULT_PATHS = [
    *(PROJECT_ROOT / path for path in (
        "data/sft/agent_traces_sft.jsonl",
        "data/sft/swebench_lite_plan_sft.jsonl",
        "data/sft/mbpp_strategy_sft.jsonl",
        "data/sft/mbpp_sft.jsonl",
        "data/sft/humaneval_sft.jsonl",
    )),
    *(PROJECT_ROOT / "data" / "llamafactory" / name for name in (
        "train_alpaca.json",
        "val_alpaca.json",
        "test_alpaca.json",
        "smoke_alpaca.json",
    )),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw JSONL and LLaMA-Factory Alpaca SFT data.")
    parser.add_argument("paths", nargs="*", type=Path, help="JSONL/JSON files or directories; defaults to project SFT data")
    parser.add_argument("--strict", action="store_true", help="Return exit code 1 when validation issues are found")
    parser.add_argument(
        "--require-explicit-task-type",
        action="store_true",
        help="Reject legacy samples whose task_type must be inferred",
    )
    parser.add_argument("--json-summary", action="store_true", help="Print the final summary as JSON")
    return parser.parse_args(argv)


def render_report(report: ValidationReport, *, json_summary: bool = False) -> None:
    for issue in report.issues:
        print(issue.render())
    summary = {
        "files_checked": report.files_checked,
        "records_checked": report.records_checked,
        "issues": len(report.issues),
        "errors": len(report.errors),
        "warnings": len(report.warnings),
        "issue_counts": report.issue_counts,
        "error_counts": report.error_counts,
        "warning_counts": report.warning_counts,
        "inferred_task_types": dict(sorted(report.inferred_task_types.items())),
        "valid": report.valid,
    }
    if json_summary:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"checked files={report.files_checked} records={report.records_checked} "
            f"errors={len(report.errors)} warnings={len(report.warnings)} valid={str(report.valid).lower()}"
        )
        if report.error_counts:
            print("error_counts=" + json.dumps(report.error_counts, ensure_ascii=False, sort_keys=True))
        if report.warning_counts:
            print("warning_counts=" + json.dumps(report.warning_counts, ensure_ascii=False, sort_keys=True))
        if report.inferred_task_types:
            print("inferred_task_types=" + json.dumps(dict(report.inferred_task_types), ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_paths(args.paths or DEFAULT_PATHS, require_explicit_task_type=args.require_explicit_task_type)
    render_report(report, json_summary=args.json_summary)
    return 1 if args.strict and not report.valid else 0


if __name__ == "__main__":
    raise SystemExit(main())
