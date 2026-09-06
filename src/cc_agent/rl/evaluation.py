from collections import Counter
import gc
import json
from pathlib import Path

from cc_agent.rl.protocol import Termination
from cc_agent.rl.rollout import rollout
from cc_agent.rl.rewards import full_success, hard_failure, test_fraction, timed_out


def metrics(results):
    if not results:
        raise ValueError("Cannot report metrics without actual rollouts")
    n = len(results)
    trajectories = [t for t, _ in results]
    observations = [o for t in trajectories for o in t.observations]
    fractions = [test_fraction(t) for t in trajectories]
    labels = ["full_success" if full_success(t) else
                       "protected_integrity_failure" if hard_failure(t) else
                       "timeout" if timed_out(t) else
                       "unverified" if not t.verification.independent or t.verification.hidden_total <= 0 or t.verification.total <= 0 else
                       "all_tests_passed_incomplete" if f == 1 else
                       "partial_test_pass" if f > 0 else "all_tests_failed"
                       for t, f in zip(trajectories, fractions)]
    outcomes = Counter(labels)
    return {
        "tasks": n,
        "full_success_rate": sum(full_success(t) for t in trajectories) / n,
        "partial_test_pass_rate": sum(0 < f < 1 for f in fractions) / n,
        "mean_test_pass_fraction": sum(fractions) / n,
        "all_tests_failed_rate": outcomes["all_tests_failed"] / n,
        "failed_finish_rate": sum(t.termination == Termination.FINISH and not full_success(t) for t in trajectories) / n,
        "max_turns_rate": sum(t.termination == Termination.MAX_TURNS for t in trajectories) / n,
        "timeout_rate": sum(timed_out(t) for t in trajectories) / n,
        "protected_integrity_failure_rate": sum(hard_failure(t) for t in trajectories) / n,
        "empty_edit_rate": sum(not t.changed for t in trajectories) / n,
        "empty_edit_attempt_rate": sum(any(o.reason == "empty_edit" for o in t.observations) for t in trajectories) / n,
        "outcome_counts": dict(outcomes),
        "outcome_termination_counts": dict(Counter(f"{label}/{t.termination.value}"
                                                  for label, t in zip(labels, trajectories))),
        "termination_counts": dict(Counter(t.termination.value for t in trajectories)),
        "task_success_rate": sum(full_success(t) for t in trajectories) / n,
        "test_pass_rate": sum(t.verification.passed for t in trajectories) / max(1, sum(t.verification.total for t in trajectories)),
        "tool_legality_rate": sum(o.legal for o in observations) / max(1, len(observations) + sum(t.format_errors for t in trajectories)),
        "average_turns": sum(len(t.actions) + t.format_errors for t in trajectories) / n,
        "average_model_tokens": sum(t.model_tokens for t in trajectories) / n,
        "truncation_rate": sum(t.termination == Termination.TOKEN_LIMIT for t in trajectories) / n,
        "reward_components": {k: sum(r.components[k] for _, r in results) / n for k in results[0][1].components},
        "total_reward": sum(r.total for _, r in results) / n,
        "failure_types": dict(Counter(("protected_integrity_failure" if hard_failure(t) else "timeout" if timed_out(t) else
                                         t.termination.value if t.termination != Termination.FINISH else "failed_finish")
                                       for t in trajectories if not full_success(t))),
    }


def evaluate(tasks, config, model_path, adapters, output):
    import torch
    from transformers import set_seed
    from cc_agent.rl.training import load_model, ModelPolicy
    output = Path(output)
    summaries = {}
    for label, adapter in adapters.items():
        set_seed(config.seed)
        model, processor = load_model(model_path, adapter, trainable=False)
        model.eval()
        results = [rollout(t, ModelPolicy(model, processor.tokenizer, sample=False), processor.tokenizer,
                           config, trace_path=output / f"{label}_rollouts.jsonl") for t in tasks]
        summaries[label] = metrics(results)
        del model, processor, results
        gc.collect()
        torch.cuda.empty_cache()
    (output / "metrics.json").write_text(json.dumps({"mock": False, "split": tasks[0].split, "results": summaries}, indent=2) + "\n")
    return summaries
