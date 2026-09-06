"""Independent CPU canaries: scripted actions, real pytest, no model downloads."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from cc_agent.rl.artifacts import adapter_check
from cc_agent.rl.config import RLConfig
from cc_agent.rl.data import load_tasks, grouped_split
from cc_agent.rl.environment import Environment, SandboxVerifier, sandbox_capability
from cc_agent.rl.hidden import partition_tests
from cc_agent.rl.protocol import Observation, Termination, Trajectory, Verification
from cc_agent.rl.rewards import score
from cc_agent.rl.rollout import ByteTokenizer, MockPolicy, rollout, task_prompt
from cc_agent.rl.training import train, audit_trainable_adapter

ROOT = Path(__file__).resolve().parents[1]


def action(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


class HardeningTests(unittest.TestCase):
    def setUp(self):
        self.task = next(t for t in load_tasks(ROOT, 'train')[0] if t.task_id == 'mbpp_27')
        self.solution = (Path(self.task.repo) / 'solution.py').read_text()
        self.config = RLConfig(num_generations=2, max_context_tokens=16000)
        self.actions = [action('read_file', path='solution.py'),
                        action('write_file', path='solution.py', content=self.solution),
                        action('run_tests'), action('finish', summary='done')]

    def test_canary_legal_real_public_and_hidden_pytest(self):
        t, reward = rollout(self.task, MockPolicy(self.actions), ByteTokenizer(), self.config)
        self.assertEqual(t.termination, Termination.FINISH)
        self.assertTrue(t.observations[2].ok)
        self.assertEqual(t.observations[2].output, 'Public tests: 1/1')
        self.assertTrue(t.verification.success)
        self.assertTrue(t.verification.independent)
        self.assertEqual(t.verification.hidden_total, 1)
        self.assertEqual(t.verification.total, 2)
        self.assertEqual(reward.components['success'], 2)
        self.assertGreater(reward.total, 0)
        self.assertEqual(t.model_tokens, sum(len(a.encode()) for a in self.actions))

    def test_canary_attack_cannot_offset_negative_reward(self):
        attacks = [action('write_file', path='tests/test_solution.py', content='assert True'),
                   action('replace_in_file', path='tests/test_solution.py', old='240', new='0'),
                   action('delete_file', path='tests/test_solution.py'),
                   action('write_file', path='conftest.py', content='pass'),
                   action('run_tests', command='pytest -q || true'),
                   action('run_tests', command=''),
                   action('run_tests', output='100 passed', returncode=0),
                   action('write_file', path='solution.py', content='print("100 passed")'),
                   action('write_file', path='solution.py', content="open('/report/result.xml', 'w').write('passed')")]
        for attack in attacks:
            with self.subTest(attack=attack):
                # First legitimately solve and pass PUBLIC tests, then attack. No prior positive
                # shaping reward or a later finish can cancel the hard failure.
                t, reward = rollout(self.task, MockPolicy(self.actions[:3] + [attack, self.actions[-1]]),
                                    ByteTokenizer(), self.config)
                self.assertEqual(t.termination, Termination.PROTECTED)
                self.assertEqual(len(t.actions), 4)
                self.assertTrue(t.observations[-1].blocked)
                self.assertFalse(t.verification.success)
                self.assertLess(reward.total, 0)
                self.assertEqual(t.completion_ids[-1], ByteTokenizer.eos_token_id)
                self.assertGreater(t.model_tokens, 0)

    def test_public_overfit_does_not_earn_success(self):
        source = self.solution.replace("return [", "return [] if list[0] == '28Jan' else [")
        actions = [action('write_file', path='solution.py', content=source), *self.actions[2:]]
        t, reward = rollout(self.task, MockPolicy(actions), ByteTokenizer(), self.config)
        self.assertTrue(t.observations[1].ok)
        self.assertFalse(t.verification.success)
        self.assertLess(reward.total, 0)

    def test_no_independent_validation_no_success(self):
        t = Trajectory('forged', changed=True, tested_current_edit=True,
                       termination=Termination.FINISH, verification=Verification(1, 1))
        self.assertLess(score(t).total, 0)
        t.verification = Verification(1, 1, independent=True, hidden_total=1)
        t.observations = [Observation(True, 'ok')] * 100 + [Observation(False, 'attack', blocked=True)]
        self.assertLess(score(t).total, 0)

    def test_hidden_assertions_absent_from_every_actor_channel(self):
        partition = partition_tests(self.task.repo)
        hidden = "remove(['28Jan', '12Jan', '11Jan']) == ['Jan', 'Jan', 'Jan']"
        seen = []
        class Spy(SandboxVerifier):
            def __call__(self, repo, timeout):
                seen.append((Path(repo) / 'tests/test_hidden.py').exists())
                return super().__call__(repo, timeout)
        with Environment(self.task, verifier=Spy()) as env:
            self.assertNotIn(hidden, json.dumps(task_prompt(self.task)))
            for p in env.root.rglob('*'):
                if p.is_file():
                    self.assertNotIn(hidden, p.read_text())
            for tool, args in [('read_file', {'path': 'tests/test_solution.py'}),
                               ('grep', {'pattern': '.*'}), ('retrieve_context', {'query': 'remove'}),
                               ('list_files', {})]:
                self.assertNotIn(hidden, env.run(tool, args).output)
            env.run('run_tests', {})
            self.assertEqual(seen, [False])
            env.verify()
            self.assertEqual(seen, [False, True])
        self.assertNotIn(hidden, partition.public)
        self.assertIn(hidden, partition.hidden)

    def test_all_partitions_use_exact_original_assertions(self):
        count = 0
        for t in load_tasks(ROOT)[0]:
            part = partition_tests(t.repo)
            def assertions(text):
                return sorted(ast.dump(n) for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Assert))
            original = (Path(t.repo) / 'tests/test_solution.py').read_text()
            self.assertEqual(assertions(original), sorted(assertions(part.public) + assertions(part.hidden)))
            self.assertFalse(set(assertions(part.public)) & set(assertions(part.hidden)))
            self.assertGreater(part.public_count, 0)
            self.assertGreater(part.hidden_count, 0)
            count += part.hidden_count
        self.assertEqual(count, 364)

    def test_train_loader_does_not_read_heldout_or_sft_data(self):
        read_text, read_bytes = Path.read_text, Path.read_bytes
        def guard(path):
            value = str(path)
            self.assertNotIn('data/llamafactory', value)
            self.assertNotIn('data/tasks', value)
            self.assertFalse(value.endswith(('/validation.jsonl', '/test.jsonl')))
            for t in load_tasks_cache:
                if t.split != 'train':
                    self.assertNotIn(t.repo + '/', value)
        load_tasks_cache = load_tasks(ROOT)[0]
        def text(path, *a, **kw):
            guard(path)
            return read_text(path, *a, **kw)
        def binary(path, *a, **kw):
            guard(path)
            return read_bytes(path, *a, **kw)
        with patch.object(Path, 'read_text', text), patch.object(Path, 'read_bytes', binary):
            tasks, _ = load_tasks(ROOT, 'train')
        self.assertTrue(all(t.split == 'train' for t in tasks))

    def test_grouped_rebalance_is_seeded_and_disjoint(self):
        tasks = load_tasks(ROOT)[0]
        a, b = grouped_split(tasks), grouped_split(list(reversed(tasks)))
        self.assertEqual({t.group_id: t.split for t in a}, {t.group_id: t.split for t in b})
        self.assertEqual([sum(t.split == s for t in a) for s in ('train', 'validation', 'test')], [151, 19, 19])
        # A balanced RL split alone would contaminate evaluation with SFT train groups.
        source_train = {t.group_id for t in tasks if t.split == 'train'}
        self.assertTrue(any(t.group_id in source_train and t.split != 'train' for t in a))

    def test_lightweight_pytest_is_real_and_bwrap_failure_detected(self):
        with patch('shutil.which', return_value=None):
            sandbox_capability.cache_clear()
            self.assertEqual(sandbox_capability()[0], 'lightweight')
        sandbox_capability.cache_clear()
        with Environment(self.task, verifier=SandboxVerifier(backend='lightweight')) as env:
            env.run('write_file', {'path': 'solution.py', 'content': self.solution})
            self.assertTrue(env.run('run_tests', {}).ok)
            self.assertTrue(env.verify().success)
        with patch('shutil.which', return_value='bwrap'), patch('subprocess.run', return_value=types.SimpleNamespace(returncode=1, stderr='namespace denied')):
            sandbox_capability.cache_clear()
            self.assertEqual(sandbox_capability()[0], 'lightweight')
        sandbox_capability.cache_clear()

    def test_adapter_audit_rejects_stacked_frozen_visual(self):
        def model(names, adapters=None, active=None):
            return types.SimpleNamespace(peft_config=adapters or {'default': {}}, active_adapters=active or ['default'],
                                         named_parameters=lambda: [(name, types.SimpleNamespace(requires_grad=trainable)) for name, trainable in names])
        valid = [('base.language_model.layer.lora_A.default.weight', True)]
        self.assertEqual(audit_trainable_adapter(model(valid)), [valid[0][0]])
        for candidate in (model([(valid[0][0], False)]), model(valid, {'default': {}, 'rl': {}}),
                          model([('base.visual.lora_A.default.weight', True)]), model([('base.language_model.weight', True)])):
            with self.assertRaises(RuntimeError):
                audit_trainable_adapter(candidate)

    def test_formal_adapter_header_and_provenance_guards(self):
        import struct
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = {"peft_type": "LORA", "target_modules": ["q_proj"]}
            def weights(name):
                header = json.dumps({name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
                (path / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
            weights("base.language_model.lora_A.weight")
            (path / "adapter_config.json").write_text(json.dumps(config))
            index = json.loads((ROOT / "data/rl/manifest.json").read_text())
            manifest = {"run_kind": "formal", "training": {"freeze_vision_tower": True, "freeze_multi_modal_projector": True},
                        "artifacts": {"datasets": {"data/llamafactory/train_alpaca.json": index["sft_train_sha256"]}}}
            (path / "experiment_manifest.json").write_text(json.dumps(manifest))
            (path / "trainer_state.json").write_text(json.dumps({"global_step": 10, "epoch": 1}))
            self.assertEqual(adapter_check(path, formal=True), path)
            weights("base.visual.lora_A.weight")
            with self.assertRaises(ValueError):
                adapter_check(path, formal=True)
            weights("base.language_model.lora_A.weight")
            manifest['run_kind'] = 'smoke'
            (path / "experiment_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                adapter_check(path, formal=True)
            manifest['run_kind'] = 'formal'
            manifest['artifacts']['datasets']['data/llamafactory/train_alpaca.json'] = 'different-split'
            (path / "experiment_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                adapter_check(path, formal=True)

    def test_training_entry_calls_bridge_with_real_environment(self):
        captured = {}
        model = types.SimpleNamespace(training=True)
        model.eval = lambda: setattr(model, 'training', False)
        model.train = lambda flag: setattr(model, 'training', flag)
        model.peft_config, model.active_adapters = {'default': {}}, ['default']
        model.named_parameters = lambda: [('base.language_model.lora_A.default.weight', types.SimpleNamespace(requires_grad=True))]
        tokenizer = ByteTokenizer()
        class Trainer:
            def __init__(inner, **kwargs):
                captured.update(kwargs)
                inner.model = kwargs['model']
                inner.accelerator = types.SimpleNamespace(unwrap_model=lambda m: m)
            def train(inner, **kwargs):
                bridge = captured['rollout_func']
                bridge.policy_factory = lambda: MockPolicy(self.actions)
                results = bridge([captured['train_dataset'][0]['prompt']] * 2, inner)
                captured['result'] = results
                captured['rewards'] = captured['reward_funcs'](['', ''], **{'verified_reward': results['verified_reward']})
            def save_model(inner, path):
                captured['saved'] = path
            def save_state(inner):
                pass
        modules = {'datasets': types.SimpleNamespace(Dataset=types.SimpleNamespace(from_list=lambda x: x)),
                   'trl': types.SimpleNamespace(GRPOConfig=lambda **kw: kw, GRPOTrainer=Trainer),
                   'transformers': types.SimpleNamespace(TrainerCallback=object)}
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', modules), \
             patch('cc_agent.rl.training.check_trl_api'), \
             patch('cc_agent.rl.training.load_model', return_value=(model, types.SimpleNamespace(tokenizer=tokenizer))), \
             patch('cc_agent.rl.training.trainable_digest', side_effect=['initial', 'updated']):
            train([self.task], self.config, 'local-model', 'formal-adapter', directory)
            rows = [json.loads(line) for line in (Path(directory) / 'rollouts.jsonl').read_text().splitlines()]
            selected = [r['payload'] for r in rows if r['event'] == 'rl_group_attempt'
                        and r['payload']['selection'] == 'accepted']
            self.assertTrue(all(r > 0 for r in selected[0]['diagnostic_total_reward']))
            self.assertTrue(all(o['full_success'] for o in selected[0]['primary_outcomes']))
        self.assertEqual(captured['args']['num_iterations'], 2)
        self.assertEqual(captured['args']['loss_type'], 'dapo')
        self.assertNotIn('peft_config', captured)
        self.assertNotIn('eval_dataset', captured)
        self.assertEqual(captured['rewards'], [0.0, 0.0])
        for ids, mask in zip(captured['result']['completion_ids'], captured['result']['env_mask']):
            self.assertEqual(len(ids), len(mask))
            self.assertIn(0, mask)
            self.assertEqual(sum(mask), sum(len(a.encode()) for a in self.actions))
        with self.assertRaises(ValueError):
            train([replace(self.task, split='validation')], self.config, '', '', '')


if __name__ == '__main__':
    unittest.main()
