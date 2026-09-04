from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_sft_environment import DATA_FILES, group_overlap
from scripts.eval_before_after_sft import (
    GenerationSettings,
    TEST_DATA_PATH,
    assert_identical_generation_contract,
    ensure_formal_adapter,
    generation_contract,
    load_test_samples,
    save_results,
    score_response,
    summarize_pairs,
)


def sample(task_type: str, output: dict, group_id: str = "fixture:1") -> dict:
    return {
        "task_type": task_type,
        "group_id": group_id,
        "instruction": "Return the required JSON.",
        "input": '{"task": "unit-test fixture"}',
        "output": json.dumps(output),
    }


class SFTEvaluationTests(unittest.TestCase):
    def test_evaluation_is_locked_to_test_and_splits_do_not_overlap(self) -> None:
        rows = load_test_samples()
        self.assertEqual(len(rows), 36)
        self.assertTrue(all(row["group_id"] for row in rows))
        self.assertTrue(all(not values for values in group_overlap(DATA_FILES[:3]).values()))
        with self.assertRaises(ValueError):
            load_test_samples(TEST_DATA_PATH.with_name("val_alpaca.json"))

    def test_tool_call_metrics(self) -> None:
        row = sample("tool_call", {"tool": "read_file", "arguments": {"path": "README.md"}})
        result = score_response(row, row["output"])
        self.assertEqual(result["metrics"]["generation_success_rate"], 1.0)
        self.assertEqual(result["metrics"]["json_valid_rate"], 1.0)
        self.assertEqual(result["metrics"]["protocol_valid_rate"], 1.0)
        self.assertEqual(result["metrics"]["tool_name_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["tool_arguments_valid_rate"], 1.0)

    def test_strategy_and_plan_metrics(self) -> None:
        strategy = {"strategy": [{"tool": "read_file", "purpose": "inspect"}, {"tool": "run_tests", "purpose": "verify"}]}
        strategy_result = score_response(sample("tool_strategy", strategy), json.dumps(strategy))
        self.assertEqual(strategy_result["metrics"]["tool_sequence_valid_rate"], 1.0)
        self.assertEqual(strategy_result["metrics"]["tool_sequence_match_rate"], 1.0)

        plan = {"plan": "Modify src/pkg/example.py with a minimal fix.", "validation": "Run focused tests."}
        plan_result = score_response(sample("swebench_plan", plan), json.dumps(plan))
        self.assertEqual(plan_result["metrics"]["plan_structure_valid_rate"], 1.0)
        self.assertEqual(plan_result["metrics"]["target_file_hit_rate"], 1.0)

    def test_invalid_output_records_failure_types(self) -> None:
        row = sample("tool_call", {"tool": "read_file", "arguments": {"path": "README.md"}})
        result = score_response(row, "not json")
        self.assertEqual(result["metrics"]["json_valid_rate"], 0.0)
        self.assertEqual(result["metrics"]["protocol_valid_rate"], 0.0)
        self.assertIn("invalid_json", result["errors"])

    def test_base_and_sft_generation_parameters_are_identical(self) -> None:
        contract = generation_contract(GenerationSettings())
        self.assertEqual(contract["base"], contract["sft"])
        self.assertFalse(contract["base"]["enable_thinking"])
        assert_identical_generation_contract(contract)
        contract["sft"]["seed"] = 7
        with self.assertRaises(ValueError):
            assert_identical_generation_contract(contract)

    def test_smoke_adapter_is_rejected_for_formal_evaluation(self) -> None:
        with self.assertRaises(ValueError):
            ensure_formal_adapter(Path("outputs/qwen3_5_2b_smoke_lora"))

    def test_results_are_written_as_jsonl_json_and_markdown(self) -> None:
        row = sample("tool_call", {"tool": "read_file", "arguments": {"path": "README.md"}})
        base = score_response(row, row["output"])
        sft = score_response(row, row["output"])
        pair = {"sample_id": row["group_id"], "sample_index": 0, "task_type": row["task_type"], "reference": row["output"], "base": base, "sft": sft}
        pairs = [pair]
        metadata = {
            "dataset_name": "coding_agent_test",
            "sample_count": 1,
            "generation": generation_contract(GenerationSettings()),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            save_results(output, pairs, summarize_pairs(pairs), metadata)
            self.assertEqual(len((output / "sample_results.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["metadata"]["dataset_name"], "coding_agent_test")
            self.assertIn("Overall metrics", (output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
