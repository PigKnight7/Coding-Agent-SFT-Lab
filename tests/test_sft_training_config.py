from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_sft_environment import (
    CONFIG_PATHS,
    DATA_FILES,
    EXPECTED_ARCHITECTURE,
    EXPECTED_MODEL_ID,
    EXPECTED_MODEL_TYPE,
    EXPECTED_TEMPLATE,
    group_overlap,
    load_config,
    static_checks,
    summarize_trainable_parameters,
    validate_config_contract,
)
from scripts.create_experiment_manifest import build_manifest, sha256_file, write_manifest
from scripts.verify_sft_adapter import latest_adapter_path, verify_training_artifacts


class SFTTrainingConfigTests(unittest.TestCase):
    def test_smoke_config_contract(self) -> None:
        config = load_config(CONFIG_PATHS[0])
        self.assertEqual(validate_config_contract(config, smoke=True), [])
        self.assertEqual(config["dataset"], "coding_agent_smoke")
        self.assertEqual(config["max_steps"], 2)
        self.assertFalse(config["train_on_prompt"])
        self.assertTrue(config["freeze_vision_tower"])
        self.assertTrue(config["freeze_multi_modal_projector"])
        self.assertFalse(config["freeze_language_model"])

    def test_formal_config_contract(self) -> None:
        config = load_config(CONFIG_PATHS[1])
        self.assertEqual(validate_config_contract(config, smoke=False), [])
        self.assertEqual(config["dataset"], "coding_agent_train")
        self.assertEqual(config["eval_dataset"], "coding_agent_val")
        self.assertEqual(config["num_train_epochs"], 1.0)
        self.assertFalse(config["train_on_prompt"])
        self.assertNotIn("coding_agent_test", {config["dataset"], config["eval_dataset"]})

    def test_template_and_model_contract_are_explicit(self) -> None:
        self.assertEqual(EXPECTED_MODEL_ID, "Qwen/Qwen3.5-2B")
        self.assertEqual(EXPECTED_MODEL_TYPE, "qwen3_5")
        self.assertEqual(EXPECTED_ARCHITECTURE, "Qwen3_5ForConditionalGeneration")
        self.assertEqual(EXPECTED_TEMPLATE, "qwen3_5_nothink")
        for path in CONFIG_PATHS:
            config = load_config(path)
            self.assertEqual(config["template"], EXPECTED_TEMPLATE)
            self.assertFalse(config["enable_thinking"])

    def test_outputs_are_independent(self) -> None:
        smoke = load_config(CONFIG_PATHS[0])
        formal = load_config(CONFIG_PATHS[1])
        self.assertNotEqual(smoke["output_dir"], formal["output_dir"])
        self.assertNotIn("adapter_name_or_path", formal)

    def test_launchers_preflight_without_hardcoded_conda(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("run_qwen3_5_2b_smoke.sh", "run_qwen3_5_2b_sft.sh"):
            text = (root / "scripts" / name).read_text(encoding="utf-8")
            self.assertNotIn("conda run", text)
            self.assertLess(text.index("check_sft_environment.py"), text.index('"${LLAMAFACTORY_CLI}" train'))
            self.assertLess(text.index("create_experiment_manifest.py"), text.index('"${LLAMAFACTORY_CLI}" train'))
            self.assertIn("overwrite_output_dir=true", text)

    def test_final_preflight_contains_all_cpu_and_runtime_gates(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts" / "final_preflight.sh").read_text(encoding="utf-8")
        for required in (
            "check_sft_environment.py",
            "validate_sft_data.py --strict --require-explicit-task-type",
            "unittest discover",
            "py_compile",
            "bash -n",
            "git diff --check",
            "CPU_ONLY",
        ):
            self.assertIn(required, text)

    def test_current_splits_have_no_group_overlap(self) -> None:
        overlaps = group_overlap(DATA_FILES[:3])
        self.assertTrue(overlaps)
        self.assertTrue(all(not values for values in overlaps.values()))

    def test_cpu_static_preflight_passes(self) -> None:
        results = static_checks(CONFIG_PATHS[0])
        failures = [result for result in results if result.status == "FAIL"]
        self.assertEqual(failures, [])

    def test_latest_adapter_path_selects_highest_valid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "checkpoint-2").mkdir()
            (root / "checkpoint-2" / "adapter_config.json").touch()
            (root / "checkpoint-10").mkdir()
            (root / "checkpoint-10" / "adapter_config.json").touch()
            (root / "checkpoint-bad").mkdir()
            self.assertEqual(latest_adapter_path(root), root / "checkpoint-10")

    def test_smoke_artifact_gate_requires_steps_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "checkpoint-2"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "adapter_model.safetensors").touch()
            (checkpoint / "trainer_state.json").write_text('{"global_step": 2, "log_history": [{"loss": 1.0}]}', encoding="utf-8")
            selected, state = verify_training_artifacts(root)
            self.assertEqual(selected, checkpoint)
            self.assertEqual(state["global_step"], 2)

    def test_trainable_parameter_audit_detects_visual_lora(self) -> None:
        class Parameter:
            def __init__(self, count: int, trainable: bool) -> None:
                self._count = count
                self.requires_grad = trainable

            def numel(self) -> int:
                return self._count

        class Model:
            def named_parameters(self):
                return [
                    ("base_model.model.language_model.layers.0.q_proj.lora_A.weight", Parameter(8, True)),
                    ("base_model.model.visual.blocks.0.attn.qkv.lora_A.weight", Parameter(8, True)),
                    ("base_model.model.visual.patch_embed.weight", Parameter(100, False)),
                ]

        summary = summarize_trainable_parameters(Model())
        self.assertEqual(summary["trainable_parameters"], 16)
        self.assertEqual(len(summary["visual_trainable_names"]), 1)

    def test_experiment_manifest_records_reproducibility_fields(self) -> None:
        config_path = CONFIG_PATHS[0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "run"
            model_path = root / "model"
            manifest = build_manifest(config_path, model_path, output, "smoke")
            self.assertEqual(manifest["run_kind"], "smoke")
            self.assertEqual(
                manifest["artifacts"]["configs"]["configs/qwen3_5_2b_smoke.yaml"],
                sha256_file(config_path),
            )
            self.assertIn("configs/qwen3_5_2b_lora_sft.yaml", manifest["artifacts"]["configs"])
            self.assertEqual(set(manifest["artifacts"]["datasets"]), {
                "data/llamafactory/train_alpaca.json",
                "data/llamafactory/val_alpaca.json",
                "data/llamafactory/test_alpaca.json",
                "data/llamafactory/smoke_alpaca.json",
                "data/llamafactory/dataset_info.json",
                "data/llamafactory/dataset_stats.json",
            })
            self.assertIn("commit", manifest["git"])
            self.assertIn("dependencies", manifest["runtime"])
            self.assertEqual(manifest["training"]["seed"], 42)
            destination = write_manifest(config_path, model_path, output, "smoke")
            saved = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(saved["output_dir"], str(output.resolve()))


if __name__ == "__main__":
    unittest.main()
