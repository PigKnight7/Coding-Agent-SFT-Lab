from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cc_agent.sft_validation import (
    infer_task_type,
    registered_tool_names,
    validate_paths,
    validate_record,
    validate_unified_diff,
)
from cc_agent.data_builder import local_tasks_to_sft, recover_task_from_plan
from scripts.prepare_llamafactory_sft import (
    assert_no_group_overlap,
    clean_record,
    convert_record,
    derive_group_id,
    repair_humaneval_code,
    split_by_group,
)


class SFTValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def valid_tool_record() -> dict:
        return {
            "task_type": "tool_call",
            "instruction": "Write a valid Python file.",
            "input": {"task": "Implement add."},
            "output": {
                "tool": "write_file",
                "arguments": {"path": "solution.py", "content": "def add(a, b):\n    return a + b\n"},
                "reason": "Implement the requested function.",
            },
        }

    def test_valid_tool_record_uses_registry(self) -> None:
        self.assertIn("write_file", registered_tool_names())
        self.assertEqual(validate_record(self.valid_tool_record()), [])

    def test_legacy_record_infers_task_type(self) -> None:
        record = self.valid_tool_record()
        del record["task_type"]
        self.assertEqual(infer_task_type(record), "tool_call")
        self.assertEqual(validate_record(record), [])
        issue_types = {issue.error_type for issue in validate_record(record, require_explicit_task_type=True)}
        self.assertIn("missing_task_type", issue_types)

    def test_reports_required_empty_unknown_tool_and_bad_arguments(self) -> None:
        record = {
            "task_type": "tool_call",
            "instruction": "",
            "input": {"task": "Unknown task"},
            "output": {"tool": "shell", "arguments": []},
        }
        issues = validate_record(record, filename="bad.jsonl", line=7)
        issue_types = {issue.error_type for issue in issues}
        self.assertTrue({"empty_field", "unknown_task", "invalid_tool"}.issubset(issue_types))
        self.assertTrue(all(issue.filename == "bad.jsonl" and issue.line == 7 for issue in issues))

    def test_rejects_invalid_python_written_by_write_file(self) -> None:
        record = self.valid_tool_record()
        record["output"]["arguments"]["content"] = "def broken(:\n    pass\n"
        self.assertIn("invalid_python_syntax", {issue.error_type for issue in validate_record(record)})

    def test_rejects_invalid_tool_arguments(self) -> None:
        record = self.valid_tool_record()
        record["output"] = {"tool": "retrieve_context", "arguments": {"query": "", "top_k": 0}}
        issue_types = [issue.error_type for issue in validate_record(record)]
        self.assertEqual(issue_types.count("invalid_tool_arguments"), 2)

        record["output"] = {"tool": "run_tests", "arguments": {"command": "rm -rf build"}}
        self.assertIn("invalid_tool_arguments", {issue.error_type for issue in validate_record(record)})

    def test_rejects_json_encoded_empty_alpaca_fields(self) -> None:
        record = self.valid_tool_record()
        record["input"] = "{}"
        self.assertIn("empty_field", {issue.error_type for issue in validate_record(record)})

    def test_validates_patch_protocol(self) -> None:
        patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
        self.assertIsNone(validate_unified_diff(patch))
        record = {
            "task_type": "swebench_patch",
            "instruction": "Create a patch.",
            "input": {"task": "Fix it."},
            "output": {"patch": "--- a/a.py\n+++ b/a.py\n"},
        }
        self.assertIn("invalid_patch", {issue.error_type for issue in validate_record(record)})

    def test_detects_credentials_sensitive_files_and_local_paths(self) -> None:
        record = self.valid_tool_record()
        record["input"] = {"task": "Use sk-abcdefghijklmnopqrstuvwxyz", "repo_path": "/home/alice/project"}
        record["output"]["arguments"]["path"] = ".env"
        issue_types = {issue.error_type for issue in validate_record(record)}
        self.assertTrue({"api_key_exposure", "local_absolute_path", "sensitive_file"}.issubset(issue_types))

        record["input"] = json.dumps(record["input"])
        record["output"] = json.dumps(record["output"])
        encoded_issue_types = {issue.error_type for issue in validate_record(record)}
        self.assertTrue({"api_key_exposure", "local_absolute_path", "sensitive_file"}.issubset(encoded_issue_types))

    def test_swebench_issue_absolute_path_is_a_warning(self) -> None:
        record = {
            "task_type": "swebench_plan",
            "instruction": "Create a repair plan.",
            "input": {"problem_statement": "Traceback in /home/reporter/project/module.py"},
            "output": {"plan": "Inspect module.py", "validation": "Run tests"},
        }
        issues = validate_record(record)
        self.assertEqual([(issue.error_type, issue.severity) for issue in issues], [("local_absolute_path", "warning")])

        record["output"] = {"plan": "Inspect /home/private/worktree/module.py", "validation": "Run tests"}
        issues = validate_record(record)
        self.assertIn(("local_absolute_path", "error"), [(issue.error_type, issue.severity) for issue in issues])

    def test_repairs_humaneval_docstring_newline(self) -> None:
        malformed = 'def f(x):\n    """Return x."""    return x\n'
        repaired = repair_humaneval_code(malformed)
        compile(repaired, "solution.py", "exec")
        self.assertIn('"""\n    return x', repaired)

    def test_recovers_unknown_trace_task_and_normalizes_strategy_tool(self) -> None:
        recovered = recover_task_from_plan("读取 `calculator.py`，修复 `subtract` 函数并运行测试。")
        self.assertIsNotNone(recovered)
        self.assertNotIn("Unknown task", recovered or "")

        trace = {
            "instruction": "Choose a tool.",
            "input": {"task": "Unknown task", "plan": "读取 `calculator.py`，修复 `subtract` 函数。"},
            "output": {"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "Inspect it"},
        }
        cleaned, repairs, exclusion = clean_record(trace, Path("agent_traces_sft.jsonl"), task_ids={})
        self.assertIsNone(exclusion)
        self.assertEqual(repairs["unknown_task_recovered"], 1)
        self.assertEqual(cleaned["task_type"], "tool_call")

        strategy = {
            "instruction": "Plan tools.",
            "input": {"task": "Fix it."},
            "output": {"strategy": [{"tool": "replace_in_file 或 write_file", "purpose": "Edit"}]},
        }
        cleaned, repairs, exclusion = clean_record(strategy, Path("mbpp_strategy_sft.jsonl"), task_ids={})
        self.assertIsNone(exclusion)
        self.assertEqual(repairs["strategy_tool_normalized"], 1)
        self.assertEqual(cleaned["output"]["strategy"][0]["tool"], "replace_in_file")

    def test_group_split_is_deterministic_and_has_no_task_overlap(self) -> None:
        records = []
        for task_index in range(20):
            for step in range(2):
                record = {
                    "task_type": "tool_call",
                    "instruction": "Choose a tool.",
                    "input": {"task": f"Task {task_index}"},
                    "output": {"tool": "read_file", "arguments": {"path": f"file_{step}.py"}, "reason": "Inspect"},
                }
                record["group_id"] = derive_group_id(record)
                records.append(record)
        first = split_by_group(records, seed=42)
        second = split_by_group(records, seed=42)
        assert_no_group_overlap(first)
        self.assertEqual(
            [[row["group_id"] for row in first[name]] for name in ("train", "validation", "test")],
            [[row["group_id"] for row in second[name]] for name in ("train", "validation", "test")],
        )
        self.assertEqual({name: len(rows) for name, rows in first.items()}, {"train": 36, "validation": 2, "test": 2})

    def test_reads_jsonl_and_pretty_alpaca_with_physical_line_numbers(self) -> None:
        raw = self.tmp_path / "raw.jsonl"
        raw.write_text(json.dumps(self.valid_tool_record()) + "\n{bad json}\n", encoding="utf-8")
        alpaca_record = self.valid_tool_record()
        alpaca_record["input"] = json.dumps(alpaca_record["input"])
        alpaca_record["output"] = json.dumps(alpaca_record["output"])
        alpaca = self.tmp_path / "alpaca.json"
        alpaca.write_text(json.dumps([alpaca_record], indent=2), encoding="utf-8")

        report = validate_paths([raw, alpaca])

        self.assertEqual(report.files_checked, 2)
        self.assertEqual(report.records_checked, 2)
        self.assertEqual(report.issue_counts, {"invalid_json": 1})
        self.assertEqual(report.issues[0].line, 2)

    def test_builders_and_alpaca_conversion_preserve_task_type(self) -> None:
        manifest = self.tmp_path / "tasks.jsonl"
        output = self.tmp_path / "strategy.jsonl"
        manifest.write_text(
            json.dumps({"id": "one", "source": "local", "repo": "repo", "task": "Fix it."}) + "\n",
            encoding="utf-8",
        )
        self.assertEqual(local_tasks_to_sft(manifest, output), 1)
        raw_record = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(raw_record["task_type"], "tool_strategy")
        self.assertEqual(convert_record(raw_record, output.name)["task_type"], "tool_strategy")

    def test_strict_cli_returns_nonzero_and_prints_location(self) -> None:
        invalid = self.tmp_path / "invalid.jsonl"
        invalid.write_text("{}\n", encoding="utf-8")
        script = Path(__file__).resolve().parents[1] / "scripts" / "validate_sft_data.py"
        completed = subprocess.run(
            [sys.executable, str(script), str(invalid), "--strict"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn(f"{invalid}:1:", completed.stdout)
        self.assertIn("missing_field", completed.stdout)

    def test_strict_cli_allows_swebench_issue_path_warning(self) -> None:
        warning_only = self.tmp_path / "warning.jsonl"
        record = {
            "task_type": "swebench_plan",
            "instruction": "Create a repair plan.",
            "input": {"problem_statement": "Failure reported from /home/reporter/project/module.py"},
            "output": {"plan": "Inspect module.py", "validation": "Run tests"},
        }
        warning_only.write_text(json.dumps(record) + "\n", encoding="utf-8")
        script = Path(__file__).resolve().parents[1] / "scripts" / "validate_sft_data.py"
        completed = subprocess.run(
            [sys.executable, str(script), str(warning_only), "--strict", "--require-explicit-task-type"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("warning: local_absolute_path", completed.stdout)
        self.assertIn("errors=0 warnings=1 valid=true", completed.stdout)


if __name__ == "__main__":
    unittest.main()
