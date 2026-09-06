"""Correctness tiers with a bounded, secondary process signal (reward v2)."""
from cc_agent.rl.protocol import Reward, Termination

SUCCESS_REWARD = 2.0
FAILURE_BASE = -3.0
PROGRESS_FLOOR = 1.5
FAILED_FINISH_PENALTY = -0.5
MAX_TURNS_PENALTY = -1.25
HARD_FAILURE_PENALTY = -3.0
AUXILIARY_BUDGET = 0.05


def hard_failure(t):
    return t.termination == Termination.PROTECTED or not t.verification.intact or any(o.blocked for o in t.observations)


def timed_out(t):
    return t.termination == Termination.TOOL_TIMEOUT or t.verification.timeout


def test_fraction(t):
    v = t.verification
    if hard_failure(t) or timed_out(t) or not v.independent or v.hidden_total <= 0:
        return 0.0
    if type(v.total) is not int or type(v.passed) is not int or not 0 <= v.passed <= v.total:
        raise ValueError("Invalid independent verification counts")
    return v.passed / v.total if v.total else 0.0


def full_success(t):
    return test_fraction(t) == 1 and t.termination == Termination.FINISH and t.changed


def length_penalty(tokens, safe=1536, limit=2048, cap=1.0, success_reward=2.0):
    if not 0 <= safe < limit or not 0 <= cap <= success_reward:
        raise ValueError("Invalid length penalty bounds")
    return -min(cap, cap * max(0, tokens - safe) / (limit - safe))


def score(t, *, safe=1536, limit=2048, cap=1.0):
    obs = t.observations
    n = max(1, len(obs) + t.format_errors)
    fraction = test_fraction(t)
    success = full_success(t)
    # L1 normalization bounds the SUM of ALL process signals, including length.
    # Scaling by testcase resolution prevents shaping from reversing an adjacent
    # passed-count difference for the same task and termination category.
    auxiliary = {
        "legality": 0.1 * sum(o.legal for o in obs) / n,
        "arguments": 0.1 * sum(o.ok for o in obs) / n,
        "format": 0.1 * len(obs) / n,
        "edit": 0.2 if t.changed else -0.2,
        "invalid": -0.2 * min(5, sum(not o.legal for o in obs) + t.format_errors),
        "repeated": -0.1 * min(5, sum(o.repeated for o in obs)),
        "empty_edit": -0.1 * min(5, sum(o.reason == "empty_edit" for o in obs)),
        "length": length_penalty(t.model_tokens, safe, limit, cap),
    }
    scale = AUXILIARY_BUDGET / (max(0, t.verification.total) + 1) / max(1.0, sum(abs(v) for v in auxiliary.values()))
    return Reward({
        "success": SUCCESS_REWARD if success else FAILURE_BASE,
        "test_progress": PROGRESS_FLOOR + fraction if fraction > 0 and not success else 0.0,
        "failed_finish": FAILED_FINISH_PENALTY if t.termination == Termination.FINISH and not success else 0.0,
        "max_turns": MAX_TURNS_PENALTY if t.termination == Termination.MAX_TURNS else 0.0,
        "untested_finish": -0.5 if t.termination == Termination.FINISH and not t.tested_current_edit else 0.0,
        "timeout": -0.5 if timed_out(t) else 0.0,
        "timeout_severity": HARD_FAILURE_PENALTY if timed_out(t) else 0.0,
        "integrity": HARD_FAILURE_PENALTY if hard_failure(t) else 0.0,
        "dangerous": -0.5 * min(2, sum(o.blocked for o in obs)),
        **{key: value * scale for key, value in auxiliary.items()},
    })
