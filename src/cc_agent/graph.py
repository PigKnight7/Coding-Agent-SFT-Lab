from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from cc_agent.config import AgentConfig
from cc_agent.actions import ACTOR_SYSTEM, execute_action
from cc_agent.json_utils import extract_json_object
from cc_agent.llm import build_chat_model
from cc_agent.repo_indexer import RepoIndexer
from cc_agent.state import AgentState, ToolRecord
from cc_agent.tools import RepoTools
from cc_agent.tracing import append_trace, create_trace_file


PLANNER_SYSTEM = """You are a careful AI coding planner.
Create a concise implementation plan before tools are used.
Prefer small, safe, testable edits. Do not invent file contents without inspecting files first.
"""

REVIEWER_SYSTEM = """You are a conservative code-review subagent.
Review the tool history, diffs, and test results. Identify whether the task is actually complete.
Return a concise review with: correctness, tests, risks, and recommended follow-up.
"""

SUMMARIZER_SYSTEM = """You summarize coding-agent runs for a developer.
Be concise. Include files changed, tests run, result, review findings, and remaining risks.
"""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _history_text(history: list[ToolRecord], limit: int = 9000) -> str:
    chunks: list[str] = []
    for item in history[-12:]:
        output = item.get("output", "")
        if len(output) > 1800:
            output = output[:1800] + "\n... output truncated"
        chunks.append(
            json.dumps(
                {
                    "tool": item.get("name"),
                    "arguments": item.get("arguments"),
                    "ok": item.get("ok"),
                    "blocked": item.get("blocked", False),
                    "output": output,
                },
                ensure_ascii=False,
            )
        )
    text = "\n".join(chunks)
    return text[-limit:]


class CodingAgentGraph:
    def __init__(self, config: AgentConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.llm = build_chat_model(config)

    def build(self):
        graph = StateGraph(AgentState)
        graph.add_node("index_repo", self.index_repo)
        graph.add_node("plan", self.plan)
        graph.add_node("act", self.act)
        graph.add_node("verify", self.verify)
        graph.add_node("review", self.review)
        graph.add_node("summarize", self.summarize)

        graph.set_entry_point("index_repo")
        graph.add_edge("index_repo", "plan")
        graph.add_edge("plan", "act")
        graph.add_edge("act", "verify")
        graph.add_conditional_edges("verify", self.route_after_verify, {"act": "act", "review": "review"})
        graph.add_edge("review", "summarize")
        graph.add_edge("summarize", END)
        return graph.compile()

    def index_repo(self, state: AgentState) -> AgentState:
        repo_path = Path(state["repo_path"]).resolve()
        trace_dir = self.config.trace_dir
        if not trace_dir.is_absolute():
            trace_dir = self.project_root / trace_dir
        trace_path = create_trace_file(trace_dir)

        snapshot = RepoIndexer(repo_path).snapshot(query=state.get("task", ""))
        new_state: AgentState = {
            **state,
            "repo_context": snapshot.as_context(),
            "project_rules": snapshot.project_rules,
            "tool_history": state.get("tool_history", []),
            "steps": state.get("steps", 0),
            "max_steps": state.get("max_steps", self.config.max_steps),
            "trace_path": str(trace_path),
        }
        append_trace(trace_path, "repo_indexed", {"repo_path": str(repo_path), "tree": snapshot.tree, "task": state.get("task", "")})
        return new_state

    def plan(self, state: AgentState) -> AgentState:
        prompt = f"""
Task:
{state['task']}

Project rules:
{state.get('project_rules', '')}

Repository context:
{state.get('repo_context', '')}

Create a short plan. Mention which files should be inspected before editing.
""".strip()
        plan = self.llm.complete(PLANNER_SYSTEM, prompt)
        append_trace(state["trace_path"], "plan", {"plan": plan})
        return {**state, "plan": plan}

    def act(self, state: AgentState) -> AgentState:
        tools = RepoTools(state["repo_path"], timeout=self.config.command_timeout)
        history = state.get("tool_history", [])
        prompt = f"""
Task:
{state['task']}

Plan:
{state.get('plan', '')}

Project rules:
{state.get('project_rules', '')}

Repository context:
{state.get('repo_context', '')[:10000]}

Test command:
{state.get('test_command') or 'pytest -q'}

{tools.descriptions()}

Recent tool history:
{_history_text(history)}

Choose the next single tool call. If complete, call finish.
""".strip()
        try:
            raw = self.llm.complete(ACTOR_SYSTEM, prompt)
        except RuntimeError as exc:
            raw = f"LLM call failed: {exc}"

        try:
            action = extract_json_object(raw)
            tool_name = str(action.get("tool", ""))
            arguments = action.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
        except Exception as exc:  # noqa: BLE001 - model output is untrusted
            tool_name = "finish"
            arguments = {"summary": f"Stopped because model did not return valid JSON: {exc}. Raw output: {raw[:500]}"}
            action = {"tool": tool_name, "arguments": arguments, "reason": "invalid_json"}

        if tool_name == "run_tests" and not arguments.get("command"):
            arguments["command"] = state.get("test_command") or "pytest -q"

        _, result = execute_action(json.dumps({"tool": tool_name, "arguments": arguments}), tools)
        record: ToolRecord = {
            "name": tool_name,
            "arguments": arguments,
            "ok": result.ok,
            "output": result.output,
            "blocked": result.blocked,
            "reason": result.reason,
        }
        next_history = [*history, record]
        steps = int(state.get("steps", 0)) + 1
        done = tool_name == "finish" or steps >= int(state.get("max_steps", self.config.max_steps))

        append_trace(
            state["trace_path"],
            "tool_call",
            {"action": action, "result": record, "step": steps},
        )
        return {
            **state,
            "last_action": action,
            "tool_history": next_history,
            "steps": steps,
            "done": done,
        }

    def verify(self, state: AgentState) -> AgentState:
        history = state.get("tool_history", [])
        last = history[-1] if history else {}
        done = bool(state.get("done"))
        verification_note = "continue"

        if last.get("name") == "finish":
            done = True
            verification_note = "model_finished"
        elif int(state.get("steps", 0)) >= int(state.get("max_steps", self.config.max_steps)):
            done = True
            verification_note = "max_steps_reached"
        elif last.get("blocked"):
            verification_note = "blocked_tool_call_continue_with_safer_action"
        elif last.get("name") == "run_tests" and last.get("ok"):
            verification_note = "tests_passed_agent_may_finish_after_diff_summary"
        elif last.get("name") == "write_file" and last.get("ok"):
            verification_note = "file_written_tests_recommended"

        append_trace(
            state["trace_path"],
            "verify",
            {"done": done, "note": verification_note, "last_tool": last.get("name")},
        )
        return {**state, "done": done}

    def route_after_verify(self, state: AgentState) -> Literal["act", "review"]:
        return "review" if state.get("done") else "act"

    def review(self, state: AgentState) -> AgentState:
        prompt = f"""
Task:
{state['task']}

Plan:
{state.get('plan', '')}

Tool history:
{_history_text(state.get('tool_history', []), limit=14000)}

Review whether the task is complete and safe to hand off.
""".strip()
        review = self.llm.complete(REVIEWER_SYSTEM, prompt)
        append_trace(state["trace_path"], "review", {"review": review})
        return {**state, "review": review}

    def summarize(self, state: AgentState) -> AgentState:
        prompt = f"""
Task:
{state['task']}

Plan:
{state.get('plan', '')}

Tool history:
{_history_text(state.get('tool_history', []), limit=14000)}

Review subagent findings:
{state.get('review', 'No review generated.')}

Trace file:
{state.get('trace_path')}

Summarize the run for the user.
""".strip()
        final_answer = self.llm.complete(SUMMARIZER_SYSTEM, prompt)
        append_trace(state["trace_path"], "final_summary", {"summary": final_answer})
        return {**state, "final_answer": final_answer, "done": True}


def run_agent(
    repo_path: str | Path,
    task: str,
    test_command: str | None,
    config: AgentConfig,
    project_root: Path,
) -> AgentState:
    app = CodingAgentGraph(config=config, project_root=project_root).build()
    initial_state: AgentState = {
        "repo_path": str(Path(repo_path).resolve()),
        "task": task,
        "test_command": test_command,
        "tool_history": [],
        "steps": 0,
        "max_steps": config.max_steps,
    }
    return app.invoke(initial_state)
