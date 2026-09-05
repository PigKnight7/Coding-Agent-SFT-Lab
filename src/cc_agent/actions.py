"""Shared action boundary for interactive and RL agents."""
from cc_agent.json_utils import extract_json_object

ACTOR_SYSTEM = """You are a Claude Code style coding agent.
You must respond with exactly one JSON object and no markdown.
Choose one tool call at a time. Inspect files before editing.
Use retrieve_context or grep to find relevant code before editing.
Prefer replace_in_file for focused edits. Use write_file only when full-file replacement is safer.
Run tests after code edits when a test command is available.

JSON schema:
{
  "tool": "tool_name",
  "arguments": {"key": "value"},
  "reason": "brief reason"
}
"""


def execute_action(raw, tools, test_command="pytest -q", *, strict=False):
    if strict:
        import json
        action = json.loads(raw)
        if not isinstance(action, dict) or set(action) - {"tool", "arguments", "reason"}:
            raise ValueError("Expected one action object")
        if not isinstance(action.get("tool"), str) or not isinstance(action.get("arguments"), dict):
            raise ValueError("Expected tool string and arguments object")
    else:
        action = extract_json_object(raw)
    name = str(action.get("tool", ""))
    arguments = action.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    if name == "run_tests" and ("command" not in arguments or (not strict and not arguments.get("command"))):
        arguments = {**arguments, "command": test_command}
    return action, tools.run(name, arguments)
