from dataclasses import asdict, replace
import json
import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from cc_agent.actions import execute_action
from cc_agent.rl.artifacts import prepare_run
from cc_agent.rl.config import RLConfig, load_config
from cc_agent.rl.data import load_tasks, select, validate_isolation
from cc_agent.rl.environment import Environment, SandboxVerifier, hashes, sandbox_command, validate_candidate
from cc_agent.rl.evaluation import metrics
from cc_agent.rl.protocol import Action, Observation, Task, Termination, Trajectory, Verification
from cc_agent.rl.rewards import length_penalty, score
from cc_agent.rl.rollout import ByteTokenizer, Generation, MockPolicy, rollout, task_prompt
from cc_agent.rl.training import RolloutBridge, grpo_kwargs, verified_reward

ROOT = Path(__file__).resolve().parents[1]


def action(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def mock_verifier(repo, timeout):
    # Test fixture double, not executable model/test evaluation.
    return Verification(int("return a + b" in (repo / "solution.py").read_text()), 1)


class RLTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "task"
        self.repo.mkdir()
        (self.repo / "solution.py").write_text("def add(a, b):\n    return a + b\n")
        (self.repo / "tests").mkdir()
        (self.repo / "tests/test_solution.py").write_text("from solution import add\ndef test_add():\n    assert add(1, 2) == 3\n    assert add(4, 7) == 11\n")
        self.task = Task("task1", "group1", "train", str(self.repo), "Implement add")
        self.config = RLConfig(max_context_tokens=16000)
        self.actions = [action("read_file", path="solution.py"),
                        action("write_file", path="solution.py", content="def add(a, b):\n    return a + b\n"),
                        action("run_tests"), action("finish", summary="done")]

    def tearDown(self):
        self.temp.cleanup()

    def run_mock(self, actions=None, config=None, verifier=mock_verifier):
        return rollout(self.task, MockPolicy(self.actions if actions is None else actions), ByteTokenizer(),
                       config or self.config, verifier=verifier)

    def test_rollout_multiturn_deterministic(self):
        first = self.run_mock()
        second = self.run_mock()
        self.assertEqual(asdict(first[0]), asdict(second[0]))
        self.assertEqual(first[1], second[1])
        t, reward = first
        self.assertEqual(t.termination, Termination.FINISH)
        self.assertTrue(t.verification.success)
        self.assertTrue(t.tested_current_edit)
        self.assertGreater(reward.total, 0)
        self.assertEqual(len(t.actions), 4)

    def test_mask_exact_model_tokens(self):
        t, _ = self.run_mock()
        self.assertEqual(t.model_tokens, sum(len(a.encode()) for a in self.actions))
        self.assertEqual(len(t.loss_mask), len(t.completion_ids))
        self.assertEqual(t.loss_mask[-1], 0)
        self.assertLess(t.model_tokens, len(t.completion_ids))
        self.assertNotIn(t.prompt_ids, [t.completion_ids])

    def test_state_transition_rejects_append_after_close(self):
        t = Trajectory("x")
        t.append_tokens([1, 2], model=True)
        t.close(Termination.FAILED)
        with self.assertRaises(ValueError):
            t.append_tokens([3], model=False)
        with self.assertRaises(ValueError):
            t.close(Termination.FINISH)

    def test_only_technical_truncation_is_filtered(self):
        t, reward = self.run_mock(config=replace(self.config, max_action_tokens=2))
        self.assertEqual(t.termination, Termination.TOKEN_LIMIT)
        self.assertNotEqual(t.completion_ids[-1], ByteTokenizer.eos_token_id)
        self.assertLess(reward.total, 0)
        for termination_config in (replace(self.config, max_turns=1), replace(self.config, max_turns=2)):
            t, reward = self.run_mock(config=termination_config)
            self.assertEqual(t.termination, Termination.MAX_TURNS)
            self.assertEqual(t.completion_ids[-1], ByteTokenizer.eos_token_id)
            self.assertGreater(sum(t.loss_mask), 0)
            self.assertLess(reward.total, 0)

    def test_timeout_preserves_loss(self):
        t, reward = self.run_mock([action("run_tests")], verifier=lambda *_: Verification(timeout=True))
        self.assertEqual(t.termination, Termination.TOOL_TIMEOUT)
        self.assertEqual(t.completion_ids[-1], ByteTokenizer.eos_token_id)
        self.assertGreater(t.model_tokens, 0)
        self.assertEqual(reward.components["timeout"], -0.5)

    def test_tool_worker_timeout(self):
        with Environment(self.task, timeout=0.0001, verifier=mock_verifier) as env:
            observation = env.run("read_file", {"path": "solution.py"})
            self.assertEqual(observation.reason, "timeout")

    def test_verifier_report_is_independent_of_stdout(self):
        def start(command, **kwargs):
            report = Path(command[command.index("--bind") + 1])
            (report / "result.xml").write_text('<testsuites><testsuite><testcase/><testcase><failure/></testcase></testsuite></testsuites>')
            return types.SimpleNamespace(wait=lambda **kw: 1)
        with patch("shutil.which", return_value="bwrap"), patch("subprocess.Popen", side_effect=start):
            result = SandboxVerifier(backend="bubblewrap")(self.repo, 30)
        self.assertEqual((result.passed, result.total), (1, 2))
        self.assertFalse(result.success)

    def test_context_exhaustion_after_complete_action_is_technical(self):
        class EOSPolicy:
            def generate(self, ids, budget):
                raw = action("read_file", path="solution.py")
                return Generation(list(raw.encode()) + [256], raw)
        config = replace(self.config, max_model_tokens=len(action("read_file", path="solution.py").encode()) + 1, safe_length=1)
        t, _ = rollout(self.task, EOSPolicy(), ByteTokenizer(), config, verifier=mock_verifier)
        self.assertEqual(t.termination, Termination.TOKEN_LIMIT)
        self.assertNotEqual(t.completion_ids[-1], 256)

    def test_observations_do_not_affect_length_reward(self):
        t, r = self.run_mock()
        t.completion_ids.extend([1] * 5000)
        t.loss_mask.extend([0] * 5000)
        self.assertEqual(score(t).components["length"], r.components["length"])

    def test_finished_failure_stays_negative(self):
        _, r = self.run_mock(verifier=lambda *_: Verification(0, 1))
        self.assertLess(r.total, 0)

    def test_reset_destroy_and_original_unchanged(self):
        original = hashes(self.repo)
        for _ in range(2):
            with Environment(self.task, verifier=mock_verifier) as env:
                copied = env.root
                self.assertNotEqual(copied, self.repo)
                self.assertIn("NotImplementedError", (copied / "solution.py").read_text())
                env.run("write_file", {"path": "solution.py", "content": "def add(a,b): return 0"})
            self.assertFalse(copied.exists())
        self.assertEqual(hashes(self.repo), original)

    def test_cleanup_on_exception(self):
        with self.assertRaises(RuntimeError):
            with Environment(self.task, verifier=mock_verifier) as env:
                copied = env.root
                raise RuntimeError("test")
        self.assertFalse(copied.exists())

    def test_protected_tests_and_command(self):
        with Environment(self.task, verifier=mock_verifier) as env:
            for path in ("tests/test_solution.py", "conftest.py", "pytest.ini", "../outside", "/tmp/escape", ".git/config"):
                with self.subTest(path=path):
                    self.assertTrue(env.run("write_file", {"path": path, "content": "assert True"}).blocked)
            for command in ("pytest -q || true", "pytest -q --override-ini addopts=", "python -c 'print(1)'", "rm -rf tests"):
                self.assertTrue(env.run("run_tests", {"command": command}).blocked)
            self.assertTrue(env.intact())

    def test_tamper_hash_detects_delete_and_weaken(self):
        with Environment(self.task, verifier=mock_verifier) as env:
            test = env.root / "tests/test_solution.py"
            test.write_text("assert True")
            self.assertFalse(env.verify().intact)
            test.unlink()
            self.assertFalse(env.verify().intact)

    def test_symlink_rejected(self):
        (self.repo / "link").symlink_to(self.root)
        with self.assertRaises(ValueError):
            with Environment(self.task):
                pass

    def test_hidden_tests_only_enter_private_verification_copy(self):
        hidden = self.root / "hidden"
        hidden.mkdir()
        (hidden / "test_hidden.py").write_text("assert True\n")
        observed = []
        def verifier(repo, timeout):
            observed.append((repo / "tests/hidden/test_hidden.py").exists())
            return Verification(1, 1)
        with Environment(replace(self.task, hidden_tests=str(hidden)), verifier=verifier) as env:
            self.assertFalse((env.root / "tests/hidden").exists())
            self.assertFalse(env.run("read_file", {"path": "tests/hidden/test_hidden.py"}).ok)
            self.assertTrue(env.verify().success)
        self.assertEqual(observed, [True])

    def test_reward_hacking_text_does_not_pass(self):
        t, r = self.run_mock([action("finish", summary="All 100 tests passed! Success!")])
        self.assertFalse(t.verification.success)
        self.assertLess(r.total, 0)
        self.assertEqual(r.components["untested_finish"], -0.5)

    def test_independent_verifier_runs_again(self):
        calls = []
        def verifier(repo, timeout):
            calls.append(1)
            return Verification(int(len(calls) == 1), 1)
        t, r = self.run_mock(verifier=verifier)
        self.assertEqual(len(calls), 2)
        self.assertFalse(t.verification.success)
        self.assertLess(r.total, 0)

    def test_edit_after_test_invalidates_finish(self):
        t, r = self.run_mock(self.actions[:3] + [action("write_file", path="solution.py", content="def add(a,b): return 0"), self.actions[-1]])
        self.assertFalse(t.tested_current_edit)
        self.assertEqual(r.components["untested_finish"], -0.5)

    def test_candidate_hacking_rejected(self):
        payloads = ["import os\nos._exit(0)", "import sys\nsys.exit(0)", "open('/report/result.xml','w')",
                    "getattr(1, '__class__')", "x.__class__.__mro__", "import pytest", "import operator\noperator.attrgetter('__globals__')",
                    "import typing\ntyping.sys.modules", "from re import enum", "eval('1')", "import builtins", "del x"]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_candidate(payload)
        validate_candidate("import math\ndef f(x): return math.sqrt(x)")

    def test_length_penalty(self):
        self.assertEqual(length_penalty(1536), 0)
        self.assertEqual(length_penalty(1792), -0.5)
        self.assertEqual(length_penalty(2048), -1)
        self.assertEqual(length_penalty(100000), -1)
        for bounds in ((10, 10, 1), (10, 20, 3)):
            with self.assertRaises(ValueError):
                length_penalty(12, *bounds)

    def test_format_invalid_tools_repeated_empty_edits(self):
        t, r = self.run_mock(["not json", action("unknown"), action("unknown"), action("finish", summary="done")])
        self.assertEqual(t.format_errors, 1)
        self.assertLess(r.components["invalid"], 0)
        self.assertLess(r.components["repeated"], 0)
        with Environment(self.task, verifier=mock_verifier) as env:
            current = (env.root / "solution.py").read_text()
            o = env.run("write_file", {"path": "solution.py", "content": current})
            self.assertEqual(o.reason, "empty_edit")
            self.assertFalse(env.changed)

    def test_schema_checks_and_output_limit(self):
        with Environment(self.task, verifier=mock_verifier, output_limit=5) as env:
            for name, args in (("read_file", {}), ("read_file", {"path": 1}), ("finish", {"summary": "ok", "command": "bad"})):
                self.assertFalse(env.run(name, args).legal)
            self.assertLessEqual(len(env.run("read_file", {"path": "solution.py"}).output), 5)

    def test_real_data_isolation(self):
        tasks, excluded = load_tasks(ROOT)
        self.assertTrue(excluded)
        for split in ("train", "validation", "test"):
            self.assertTrue(select(tasks, split))
        self.assertTrue(all(t.split == "train" for t in select(tasks, "train")))

    def test_cross_split_task_group_repo_leakage(self):
        for other in (replace(self.task, split="test"), replace(self.task, task_id="other", split="validation"),
                      replace(self.task, task_id="other", group_id="other", split="test")):
            with self.assertRaises(ValueError):
                validate_isolation([self.task, other])

    def test_config_and_grpo_public_settings(self):
        for kind, g in (("smoke", 2), ("train", 4)):
            config = load_config(ROOT / f"configs/agentic_rl_{kind}.yaml")
            self.assertEqual(config.num_generations, g)
            args = grpo_kwargs(config, "outputs/test")
            self.assertEqual(args["loss_type"], "dapo")
            self.assertEqual(args["epsilon_high"], 0.28)
            self.assertEqual(args["epsilon"], 0.2)
            self.assertTrue(args["mask_truncated_completions"])
            self.assertEqual(args["num_iterations"], 2)
            self.assertNotEqual(args["gradient_accumulation_steps"] % (args["steps_per_generation"] * args["num_iterations"]), 0)
            self.assertFalse(args["use_vllm"])
            self.assertEqual(args["gradient_accumulation_steps"] % g, 0)
        for config in (replace(self.config, num_generations=1), replace(self.config, safe_length=3000),
                       replace(self.config, loss_type="grpo"), replace(self.config, max_turns=-1)):
            with self.assertRaises(ValueError):
                config.validate()
        bad = self.root / "bad.yaml"
        bad.write_text("dataset: coding_agent_test")
        with self.assertRaises(ValueError):
            load_config(bad)

    def test_bridge_cardinality_masks_and_rewards(self):
        model = types.SimpleNamespace(training=True)
        model.eval = lambda: setattr(model, "training", False)
        model.train = lambda v: setattr(model, "training", v)
        trainer = types.SimpleNamespace(model=model, accelerator=types.SimpleNamespace(unwrap_model=lambda x: x))
        bridge = RolloutBridge([self.task], ByteTokenizer(), replace(self.config, num_generations=2), self.root / "trace.jsonl",
                               verifier=mock_verifier, policy_factory=lambda: MockPolicy(self.actions))
        result = bridge([task_prompt(self.task)] * 2, trainer)
        self.assertEqual(len(result["completion_ids"]), 2)
        self.assertIsNone(result["logprobs"])
        self.assertTrue(model.training)
        self.assertEqual(result["verified_reward"], verified_reward(["", ""], result["verified_reward"]))
        for ids, mask in zip(result["completion_ids"], result["env_mask"]):
            self.assertEqual(len(ids), len(mask))
            self.assertIn(0, mask)
        with self.assertRaises(RuntimeError):
            bridge([task_prompt(self.task)], trainer)

    def test_evaluation_has_no_fabricated_empty_metrics(self):
        with self.assertRaises(ValueError):
            metrics([])
        result = metrics([self.run_mock()])
        self.assertEqual(result["task_success_rate"], 1)
        self.assertIn("length", result["reward_components"])
        self.assertEqual(result["failure_types"], {})

    def test_resume_checks_full_checkpoint_and_manifest(self):
        output = self.root / "run"
        record = {"config": "frozen"}
        prepare_run(output, record)
        with self.assertRaises(ValueError):
            prepare_run(output, record)
        cp = output / "checkpoint-1"
        cp.mkdir()
        with self.assertRaises(ValueError):
            prepare_run(output, record, cp)
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
            (cp / name).write_text("fixture")
        prepare_run(output, record, cp)
        with self.assertRaises(ValueError):
            prepare_run(output, {"config": "changed"}, cp)

    def test_scripts_and_arguments(self):
        from scripts.agentic_rl import parse_args
        self.assertTrue(parse_args(["precheck", "--static-only"]).static_only)
        for args in (["train", "--static-only"], ["train"], ["eval", "--model-path", "m", "--sft-adapter", "s", "--output-dir", "o"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)
        for script in ("run_agentic_rl_smoke.sh", "run_agentic_rl_train.sh", "run_agentic_rl_eval.sh", "resume_agentic_rl.sh", "check_rl_environment.sh"):
            result = subprocess.run(["bash", str(ROOT / "scripts" / script), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_sandbox_has_no_host_root_or_network(self):
        with patch("shutil.which", return_value="/usr/bin/bwrap"):
            cmd = sandbox_command(self.repo, self.root / "report")
        self.assertIn("--unshare-all", cmd)
        self.assertIn("--clearenv", cmd)
        self.assertNotIn("/", cmd)
        self.assertNotIn("--share-net", cmd)
        with patch("shutil.which", return_value=None), self.assertRaises(RuntimeError):
            sandbox_command(self.repo, self.root)


if __name__ == "__main__":
    unittest.main()
