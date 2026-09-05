from collections import Counter
import gc
import json
from pathlib import Path

from cc_agent.rl.protocol import Termination
from cc_agent.rl.rollout import rollout


def metrics(results):
    if not results:
        raise ValueError("Cannot report metrics without actual rollouts")
    n = len(results)
    trajectories = [t for t, _ in results]
    observations = [o for t in trajectories for o in t.observations]
    return {
        "tasks": n,
        "task_success_rate": sum(t.verification.success and t.verification.independent and t.verification.hidden_total > 0 and t.changed and t.termination == Termination.FINISH for t in trajectories) / n,
        "test_pass_rate": sum(t.verification.passed for t in trajectories) / max(1, sum(t.verification.total for t in trajectories)),
        "tool_legality_rate": sum(o.legal for o in observations) / max(1, len(observations) + sum(t.format_errors for t in trajectories)),
        "average_turns": sum(len(t.actions) + t.format_errors for t in trajectories) / n,
        "average_model_tokens": sum(t.model_tokens for t in trajectories) / n,
        "truncation_rate": sum(t.termination == Termination.TOKEN_LIMIT for t in trajectories) / n,
        "reward_components": {k: sum(r.components[k] for _, r in results) / n for k in results[0][1].components},
        "total_reward": sum(r.total for _, r in results) / n,
        "failure_types": dict(Counter((t.termination.value if t.termination != Termination.FINISH else "task_failed")
                                       for t in trajectories if not (t.verification.success and t.verification.independent and t.verification.hidden_total > 0 and t.changed and t.termination == Termination.FINISH))),
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
