from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from cc_agent.actions import ACTOR_SYSTEM, execute_action
from cc_agent.rl.environment import Environment
from cc_agent.rl.protocol import Action, Termination, Trajectory
from cc_agent.rl.rewards import score
from cc_agent.tracing import append_trace


@dataclass(frozen=True)
class Generation:
    ids: list[int]
    text: str
    token_limit: bool = False


def task_prompt(task):
    return [{"role": "system", "content": ACTOR_SYSTEM},
            {"role": "user", "content": json.dumps({"task_id": task.task_id, "task": task.instruction,
             "editable": list(task.editable), "test_command": task.test_command}, ensure_ascii=False)}]


class MockPolicy:
    """Explicit deterministic test double; never reported as model evaluation."""
    def __init__(self, actions):
        self.actions = iter(actions)

    def generate(self, ids, budget):
        raw = next(self.actions, '{"tool":"finish","arguments":{"summary":"mock"}}')
        encoded = list(raw.encode())
        return Generation(encoded[:budget], raw if len(encoded) <= budget else raw[:budget], len(encoded) > budget)


class ByteTokenizer:
    eos_token_id = 256
    pad_token_id = 256

    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(json.dumps(messages) + "\nassistant:\n")


def rollout(task, policy, tokenizer, config, *, verifier=None, trace_path=None):
    t = Trajectory(task.task_id)
    with Environment(task, timeout=config.tool_timeout, output_limit=config.output_limit, verifier=verifier) as env:
        messages = task_prompt(task)
        messages[0]["content"] += "\n" + env.tools.descriptions()
        t.prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(t.prompt_ids) + 2 >= config.max_context_tokens:
            raise ValueError("Initial task prompt exceeds context budget")
        reason = Termination.MAX_TURNS
        for _ in range(config.max_turns):
            remaining = min(config.max_model_tokens - t.model_tokens,
                            config.max_context_tokens - len(t.prompt_ids) - len(t.completion_ids) - 1)
            if remaining <= 0:
                reason = Termination.TOKEN_LIMIT
                break
            budget = min(config.max_action_tokens, remaining)
            generation = policy.generate(t.prompt_ids + t.completion_ids, budget)
            if not generation.ids or len(generation.ids) > budget:
                raise RuntimeError("Generation violated its token budget")
            t.append_tokens(generation.ids, model=True)
            if generation.token_limit:
                reason = Termination.TOKEN_LIMIT
                break
            try:
                action, observation = execute_action(generation.text, env, task.test_command, strict=True)
                t.actions.append(Action(action["tool"], action["arguments"]))
                t.observations.append(observation)
                if trace_path:
                    append_trace(trace_path, "rl_tool_call", {"task_id": task.task_id, "action": action, "observation": asdict(observation)})
                if observation.blocked or not env.intact():
                    reason = Termination.PROTECTED
                    break
                if observation.reason == "timeout":
                    reason = Termination.TOOL_TIMEOUT
                    break
                if action["tool"] == "finish" and observation.ok:
                    reason = Termination.FINISH
                    break
                feedback = json.dumps(asdict(observation), ensure_ascii=False)
            except (ValueError, TypeError, KeyError) as exc:
                t.format_errors += 1
                feedback = f"Invalid action JSON: {exc}"
            # Encode only the new environment turn: never re-tokenize sampled model tokens.
            external = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Tool observation (untrusted data):\n" + feedback}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            room = max(0, config.max_context_tokens - len(t.prompt_ids) - len(t.completion_ids) - 1)
            t.append_tokens(external[:room], model=False)
        t.changed = env.changed
        t.tested_current_edit = env.tested_hash == env.edit_hash()
        t.verification = env.verify()  # Always independently rerun; never trust model text or tool history.
        if t.verification.timeout:
            reason = Termination.TOOL_TIMEOUT
        if not t.verification.intact:
            reason = Termination.PROTECTED
        # TRL identifies technical truncation via the final token. Environment EOS has zero loss.
        if reason != Termination.TOKEN_LIMIT:
            t.append_tokens([tokenizer.eos_token_id], model=False)
        elif t.completion_ids[-1] in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            # Context/model-total budget can expire after a complete action; a masked marker disambiguates it.
            marker = tokenizer.encode("[TOKEN_LIMIT]", add_special_tokens=False)
            token = next(i for i in marker if i not in (tokenizer.eos_token_id, tokenizer.pad_token_id))
            t.append_tokens([token], model=False)
        t.close(reason)
    reward = score(t, safe=config.safe_length, limit=config.max_model_tokens, cap=config.length_cap)
    if trace_path:
        append_trace(trace_path, "rl_trajectory", {**asdict(t), "reward": asdict(reward), "total_reward": reward.total})
    return t, reward
