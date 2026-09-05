from cc_agent.rl.protocol import Reward, Termination


def length_penalty(tokens, safe=1536, limit=2048, cap=1.0, success_reward=2.0):
    if not 0 <= safe < limit or not 0 <= cap <= success_reward:
        raise ValueError("Invalid length penalty bounds")
    return -min(cap, cap * max(0, tokens - safe) / (limit - safe))


def score(t, *, safe=1536, limit=2048, cap=1.0):
    obs = t.observations
    n = max(1, len(obs) + t.format_errors)
    hard_failure = t.termination == Termination.PROTECTED or not t.verification.intact or any(o.blocked for o in obs)
    success = not hard_failure and t.verification.independent and t.verification.hidden_total > 0 and t.verification.success and t.termination == Termination.FINISH and t.changed
    return Reward({
        "success": 2.0 if success else -1.0,
        "legality": 0.1 * sum(o.legal for o in obs) / n,
        "arguments": 0.1 * sum(o.ok for o in obs) / n,
        "format": 0.1 * len(obs) / n,
        "edit": 0.2 if t.changed else -0.2,
        "invalid": -0.2 * min(5, sum(not o.legal for o in obs) + t.format_errors),
        "dangerous": -0.5 * min(2, sum(o.blocked for o in obs)),
        "repeated": -0.1 * min(5, sum(o.repeated for o in obs)),
        "empty_edit": -0.1 * min(5, sum(o.reason == "empty_edit" for o in obs)),
        "untested_finish": -0.5 if t.termination == Termination.FINISH and not t.tested_current_edit else 0.0,
        "timeout": -0.5 if t.termination == Termination.TOOL_TIMEOUT or t.verification.timeout else 0.0,
        "integrity": -2.0 if hard_failure else 0.0,
        "length": length_penalty(t.model_tokens, safe, limit, cap),
    })
