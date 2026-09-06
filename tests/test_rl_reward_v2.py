"""CPU regression contracts; scripted trajectories are not model evidence."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cc_agent.rl.config import RLConfig, load_config
from cc_agent.rl.evaluation import metrics
from cc_agent.rl.protocol import Observation, Reward, Task, Termination, Trajectory, Verification
from cc_agent.rl.rewards import score, primary_outcome, optimization_rewards
from cc_agent.rl.rollout import ByteTokenizer, task_prompt
from cc_agent.rl.training import RolloutBridge, grpo_kwargs, verified_reward
from scripts.verify_rl_run import verify_run

ROOT = Path(__file__).resolve().parents[1]


def trajectory(passed=0, total=2, termination=Termination.FINISH, *, poor=False):
    t = Trajectory('a', changed=not poor, tested_current_edit=not poor,
                   verification=Verification(passed, total, independent=True, hidden_total=1),
                   termination=termination)
    t.prompt_ids = [9]
    t.completion_ids = [1, 2, 3, 256]
    t.loss_mask = [1, 0, 1, 0]
    t.observations = [Observation(not poor, '', legal=not poor, repeated=poor,
                                  reason='empty_edit' if poor else '')]
    t.format_errors = 5 if poor else 0
    return t


class RewardV2Tests(unittest.TestCase):
    def test_success_partial_zero_terminal_hard_order_bounds(self):
        # Exhaust all combinations of shaping, current-edit testing and length caps.
        tiers = {key: [] for key in ('success', 'partial', 'zero', 'finish', 'max', 'hard')}
        for poor, tested, cap, term, passed in itertools.product(
                (False, True), (False, True), (0, 2), tuple(Termination), (0, 1, 2)):
            t = trajectory(passed, termination=term, poor=poor)
            t.changed = True
            t.tested_current_edit = tested
            t.loss_mask = [1] * 3000
            t.completion_ids = [1] * 3000
            r = score(t, cap=cap)
            if term in (Termination.PROTECTED, Termination.TOOL_TIMEOUT):
                tiers['hard'].append(r.total)
            elif passed == 2 and term == Termination.FINISH:
                tiers['success'].append(r.total)
            elif passed == 1:
                tiers['partial'].append(r.total)
            elif passed == 0:
                key = 'finish' if term == Termination.FINISH else 'max' if term == Termination.MAX_TURNS else 'zero'
                tiers[key].append(r.total)
        for higher, lower in zip(tiers, list(tiers)[1:]):
            self.assertGreater(min(tiers[higher]), max(tiers[lower]), (higher, lower))

    def test_half_pass_beats_all_fail(self):
        self.assertGreater(score(trajectory(1)).total, score(trajectory()).total + 1)

    def test_failed_finish_even_when_current_edit_tested(self):
        t = trajectory()
        self.assertTrue(t.tested_current_edit)
        self.assertEqual(score(t).components['failed_finish'], -0.5)
        self.assertEqual(score(t).components['untested_finish'], 0)

    def test_max_turns_worse_than_even_untested_failed_finish(self):
        t = trajectory()
        t.tested_current_edit = False
        self.assertLess(score(trajectory(termination=Termination.MAX_TURNS)).total, score(t).total)

    def test_legal_edit_all_failed_cannot_be_positive(self):
        for term in Termination:
            self.assertLess(score(trajectory(termination=term)).total, 0)

    def test_hard_failure_cannot_earn_progress_or_success(self):
        for condition in ('blocked', 'intact', 'timeout'):
            t = trajectory(2)
            if condition == 'blocked':
                t.observations = [Observation(False, '', blocked=True)]
            else:
                t.verification = replace(t.verification, **{condition: condition == 'timeout'})
            self.assertLess(score(t).total, -5.9)
            self.assertEqual(score(t).components['test_progress'], 0)
            self.assertEqual(metrics([(t, score(t))])['full_success_rate'], 0)

    def test_untrusted_counts_do_not_earn_progress(self):
        for independent, hidden_total in ((False, 1), (True, 0)):
            t = trajectory(2)
            t.verification = replace(t.verification, independent=independent, hidden_total=hidden_total)
            self.assertEqual(score(t).components['test_progress'], 0)
            self.assertLess(score(t).total, 0)
        t.verification = Verification(3, 2, independent=True, hidden_total=1)
        with self.assertRaises(ValueError):
            score(t)

    def test_auxiliary_cannot_reverse_adjacent_pass_counts(self):
        # Include high testcase counts, where fixed shaping weights would dominate.
        for total in (2, 3, 100, 10000):
            for passed in (0, total // 2, total - 1):
                better = trajectory(passed + 1, total, poor=True)
                worse = trajectory(passed, total)
                better.changed = True
                better.tested_current_edit = True
                self.assertGreater(score(better).total, score(worse).total)

    def test_evaluation_outcomes_and_denominators(self):
        ts = [trajectory(2), trajectory(1), trajectory(),
              trajectory(termination=Termination.MAX_TURNS),
              trajectory(termination=Termination.TOOL_TIMEOUT),
              trajectory(termination=Termination.PROTECTED)]
        ts[2].changed = False
        ts[2].observations.append(Observation(True, '', reason='empty_edit'))
        result = metrics([(t, score(t)) for t in ts])
        for key in ('task_success_rate', 'full_success_rate', 'partial_test_pass_rate',
                    'max_turns_rate', 'timeout_rate', 'protected_integrity_failure_rate',
                    'empty_edit_rate', 'empty_edit_attempt_rate'):
            self.assertEqual(result[key], 1 / 6, key)
        self.assertEqual(result['mean_test_pass_fraction'], 0.25)
        self.assertEqual(result['test_pass_rate'], 0.25)
        self.assertEqual(result['failed_finish_rate'], 2 / 6)
        self.assertEqual(result['outcome_counts']['all_tests_failed'], 2)
        self.assertEqual(result['failure_types']['failed_finish'], 2)
        self.assertEqual(result['outcome_termination_counts']['partial_test_pass/finish'], 1)
        self.assertEqual(result['outcome_termination_counts']['all_tests_failed/finish'], 1)
        self.assertEqual(result['truncation_rate'], 0)


class DynamicSamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.trace = Path(self.tmp.name) / 'rollouts.jsonl'
        self.tasks = [Task(x, x, 'train', '.', x) for x in ('a', 'b')]
        model = SimpleNamespace(training=True)
        model.eval = lambda: setattr(model, 'training', False)
        model.train = lambda flag: setattr(model, 'training', flag)
        self.trainer = SimpleNamespace(model=model, accelerator=SimpleNamespace(unwrap_model=lambda x: x, num_processes=1))

    def run_bridge(self, rewards, *, groups=1, **config):
        self.generated = []
        values = iter(rewards)
        def generate(task, *args, **kwargs):
            value = next(values)
            t = value if isinstance(value, Trajectory) else trajectory(int(value), total=10)
            t.task_id = task.task_id
            t.prompt_ids = [ord(task.task_id)]
            t.completion_ids[0] = len(self.generated) + 10
            pair = (t, score(t) if isinstance(value, Trajectory) else Reward({'fixture': value}))
            self.generated.append(pair)
            return pair
        self.bridge = RolloutBridge(self.tasks, ByteTokenizer(), RLConfig(num_generations=2, **config), self.trace)
        prompts = [task_prompt(task) for task in self.tasks[:groups] for _ in range(2)]
        with patch('cc_agent.rl.training.rollout', side_effect=generate):
            result = self.bridge(prompts, self.trainer)
        self.rows = [json.loads(s) for s in self.trace.read_text().splitlines()]
        self.attempts = [r['payload'] for r in self.rows if r['event'] == 'rl_group_attempt']
        return result

    def test_zero_group_resampled_discarded_not_returned_order_masks_exact(self):
        output = self.run_bridge([0, 0, 1, 2, 3, 4], groups=2)
        selected = self.generated[2:]
        self.assertEqual(output['verified_reward'], [1, 2, 3, 4])
        for key, attr in (('prompt_ids', 'prompt_ids'), ('completion_ids', 'completion_ids'), ('env_mask', 'loss_mask')):
            self.assertEqual(output[key], [getattr(t, attr) for t, _ in selected])
            self.assertEqual(len(output[key]), 4)
        self.assertEqual(output['prompt_ids'], [[97], [97], [98], [98]])
        self.assertIsNone(output['logprobs'])
        self.assertEqual([a['selection'] for a in self.attempts], ['discarded', 'accepted', 'accepted'])
        self.assertEqual(self.bridge.last_statistics['dynamic_sampling_retry_count'], 1)
        self.assertEqual(self.bridge.last_statistics['correctness_zero_variance_group_rate'], 0.5)
        self.assertTrue(self.trainer.model.training)
        ids = [rid for a in self.attempts for rid in a['rollout_ids']]
        self.assertEqual(len(ids), len(set(ids)))

    def test_nonzero_no_retry(self):
        self.run_bridge([0, 1])
        self.assertEqual(len(self.generated), 2)
        self.assertEqual(self.bridge.last_statistics['dynamic_sampling_retry_count'], 0)
        self.assertEqual(self.bridge.last_statistics['dynamic_sampling_exhausted_rate'], 0)

    def test_exhausted_strict_bound_last_group_kept(self):
        result = self.run_bridge([0, 0, 1, 1, 2, 2], dynamic_sampling_max_retries=2)
        self.assertEqual(len(self.generated), 6)
        self.assertEqual(result['verified_reward'], [0.0, 0.0])
        self.assertEqual(self.bridge.last_statistics['dynamic_sampling_retry_count'], 2)
        self.assertEqual(self.bridge.last_statistics['dynamic_sampling_exhausted_rate'], 1)
        self.assertTrue(self.attempts[-1]['exhausted'])

    def test_zero_retry_and_disabled_semantics(self):
        for enabled in (True, False):
            self.run_bridge([0, 0], dynamic_sampling_max_retries=0, dynamic_sampling_enabled=enabled)
            self.assertEqual(len(self.generated), 2)
            self.assertEqual(self.bridge.last_statistics['dynamic_sampling_exhausted_rate'], int(enabled))

    def test_total_precision_does_not_make_equal_outcomes_informative(self):
        self.run_bridge([1, 1 + 1e-9, 0, 1], zero_variance_epsilon=0)
        self.assertEqual(len(self.generated), 4)
        self.run_bridge([0, 1e-7, 0, 1])
        self.assertEqual(len(self.generated), 4)

    def test_auxiliary_only_variance_resamples(self):
        a, b = trajectory(), trajectory(poor=True)
        output = self.run_bridge([a, b, trajectory(), trajectory(1)])
        self.assertEqual(len(self.generated), 4)
        self.assertEqual(output['verified_reward'], [r.total for _, r in self.generated[2:]])
        self.assertTrue(self.attempts[0]['correctness_zero_variance'])
        self.assertFalse(self.attempts[0]['total_reward_zero_variance'])
        self.assertEqual(self.bridge.last_statistics['total_reward_zero_variance_group_rate'], 0)

    def test_exhausted_auxiliary_diagnostics_and_actual_float32_advantages(self):
        import torch
        a, b = trajectory(), trajectory()
        b.observations.append(Observation(True, '', repeated=True))
        output = self.run_bridge([a, b] * 3, dynamic_sampling_max_retries=2)
        self.assertEqual(len(self.generated), 6)
        diagnostics = self.attempts[-1]['diagnostic_total_reward']
        self.assertNotEqual(*diagnostics)
        self.assertEqual(self.attempts[-1]['reward_components'], [score(a).components, score(b).components])
        rewards = verified_reward(output['completion_ids'], output['verified_reward'])
        self.assertEqual(rewards, [0.0, 0.0])
        def advantages(values):
            x = torch.tensor(values, dtype=torch.float32)
            return (x - x.mean()) / (x.std() + 1e-4)
        self.assertTrue(torch.count_nonzero(advantages(diagnostics)).item() > 0)
        self.assertEqual(torch.count_nonzero(advantages(rewards)).item(), 0)
        self.assertEqual(self.bridge.last_statistics['optimization_reward_zeroed_group_rate'], 1)
        for key, attr in (('prompt_ids', 'prompt_ids'), ('completion_ids', 'completion_ids'), ('env_mask', 'loss_mask')):
            self.assertEqual(output[key], [getattr(t, attr) for t, _ in self.generated[-2:]])

    def test_safety_only_variance_preserves_penalty(self):
        for kind in ('integrity', 'protected', 'blocked', 'timeout'):
            unsafe = trajectory()
            if kind == 'integrity':
                unsafe.verification = replace(unsafe.verification, intact=False)
            elif kind == 'protected':
                unsafe.termination = Termination.PROTECTED
            elif kind == 'blocked':
                unsafe.observations.append(Observation(False, '', blocked=True))
            else:
                unsafe.verification = replace(unsafe.verification, timeout=True)
            output = self.run_bridge([trajectory(), unsafe])
            self.assertEqual(len(self.generated), 2)
            self.assertGreater(*output['verified_reward'])
            self.assertEqual(self.bridge.last_statistics['optimization_reward_zeroed_group_rate'], 0)

    def test_cross_termination_correctness_order_is_enforced(self):
        better = trajectory(51, total=100, termination=Termination.MAX_TURNS, poor=True)
        worse = trajectory(50, total=100, termination=Termination.FAILED)
        self.assertLess(score(better).total, score(worse).total)
        output = self.run_bridge([worse, better])
        self.assertLess(*output['verified_reward'])
        self.assertEqual(len(self.generated), 2)

    def test_primary_variance_survives_identical_diagnostic_totals(self):
        for better, worse in ((trajectory(1), trajectory()),
                              (trajectory(2), trajectory(2, termination=Termination.MAX_TURNS)),
                              (trajectory(), trajectory(termination=Termination.PROTECTED))):
            rewards = optimization_rewards([primary_outcome(worse), primary_outcome(better)], [1.0, 1.0])
            self.assertLess(*rewards)

    def test_disabled_and_zero_retries_still_zero_auxiliary_differences(self):
        for config in ({'dynamic_sampling_enabled': False}, {'dynamic_sampling_max_retries': 0}):
            output = self.run_bridge([trajectory(), trajectory(poor=True)], **config)
            self.assertEqual(output['verified_reward'], [0.0, 0.0])
            self.assertEqual(self.bridge.last_statistics['optimization_reward_zeroed_group_rate'], 1)
            self.assertNotEqual(*self.attempts[-1]['diagnostic_total_reward'])

    def test_untrusted_success_counts_do_not_create_variance(self):
        untrusted = trajectory(2)
        untrusted.verification = replace(untrusted.verification, independent=False)
        output = self.run_bridge([trajectory(), untrusted], dynamic_sampling_enabled=False)
        self.assertEqual(output['verified_reward'], [0.0, 0.0])

    def test_exception_restores_model_no_selected_batch(self):
        with self.assertRaises(StopIteration):
            self.run_bridge([0, 0])
        self.assertTrue(self.trainer.model.training)
        rows = [json.loads(s) for s in self.trace.read_text().splitlines()]
        self.assertFalse(any(r['event'] == 'rl_batch_selected' for r in rows))

    def test_invalid_groups_and_multiprocess_fail_closed(self):
        bridge = RolloutBridge(self.tasks, ByteTokenizer(), RLConfig(num_generations=2), self.trace)
        for prompts in ([], [task_prompt(self.tasks[0])], [task_prompt(t) for t in self.tasks]):
            with self.assertRaises(RuntimeError):
                bridge(prompts, self.trainer)
        self.trainer.accelerator.num_processes = 2
        with self.assertRaises(RuntimeError):
            bridge([task_prompt(self.tasks[0])] * 2, self.trainer)

    def test_strict_config_and_smoke_formal_semantics(self):
        for key, values in {
            'dynamic_sampling_enabled': (1, 'true', None),
            'dynamic_sampling_max_retries': (-1, 17, True, 1.2),
            'zero_variance_epsilon': (-1, float('nan'), float('inf'), True, 0.1),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    replace(RLConfig(), **{key: value}).validate()
        a, b = [load_config(ROOT / f'configs/agentic_rl_{kind}.yaml') for kind in ('smoke', 'train')]
        for key in ('dynamic_sampling_enabled', 'dynamic_sampling_max_retries', 'zero_variance_epsilon'):
            self.assertEqual(getattr(a, key), getattr(b, key))
        for config in (a, b):
            kw = grpo_kwargs(config, 'unused')
            self.assertEqual((kw['loss_type'], kw['epsilon'], kw['epsilon_high']), ('dapo', 0.2, 0.28))
            self.assertTrue(kw['mask_truncated_completions'])
            self.assertEqual(kw['num_iterations'], 2)
            self.assertNotEqual(kw['gradient_accumulation_steps'] % (kw['steps_per_generation'] * kw['num_iterations']), 0)

    def test_sampling_metrics_written_with_batch_denominator(self):
        self.run_bridge([0, 0, 0, 1])
        rows = [json.loads(s) for s in (self.trace.parent / 'training_log.jsonl').read_text().splitlines()]
        payload = rows[-1]['payload']
        self.assertEqual(payload['tasks'], 2)
        self.assertEqual(payload['group_count'], 1)
        for key in ('full_success_rate', 'partial_test_pass_rate', 'mean_test_pass_fraction',
                    'failed_finish_rate', 'max_turns_rate', 'empty_edit_rate', 'correctness_zero_variance_group_rate',
                    'accepted_correctness_zero_variance_group_rate', 'total_reward_zero_variance_group_rate',
                    'optimization_reward_zeroed_group_rate',
                    'dynamic_sampling_retry_count', 'dynamic_sampling_exhausted_rate'):
            self.assertIn(key, payload)

    def test_smoke_gate_rejects_discarded_or_uncommitted_candidates(self):
        root = self.trace.parent
        (root / 'trainer_state.json').write_text(json.dumps({'global_step': 2, 'log_history': [{'loss': 0.1}]}))
        (root / 'final_adapter').mkdir()
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            (root / 'final_adapter' / name).write_text('fixture')
        (root / 'model_audit.json').write_text(json.dumps({'weights_changed': True,
            'gradient_steps': [{'nonzero': True}], 'grpo_kwargs': grpo_kwargs(RLConfig(), root)}))
        candidate = {'event': 'rl_trajectory', 'payload': {'rollout_id': 'x', 'actions': [{}, {}],
            'observations': [{'legal': True}], 'loss_mask': [1, 0], 'completion_ids': [1, 256]}}
        attempt = {'event': 'rl_group_attempt', 'payload': {'selection': 'discarded', 'batch_id': 'b', 'rollout_ids': ['x']}}
        batch = {'event': 'rl_batch_selected', 'payload': {'batch_id': 'b'}}
        for status, committed in (('discarded', True), ('accepted', False), ('accepted', True)):
            attempt['payload']['selection'] = status
            rows = [candidate, attempt] + ([batch] if committed else [])
            self.trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            if status == 'accepted' and committed:
                self.assertEqual(verify_run(root, 2)['trajectories'], 1)
            else:
                with self.assertRaises(ValueError):
                    verify_run(root, 2)


if __name__ == '__main__':
    unittest.main()
