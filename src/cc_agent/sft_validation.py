from __future__ import annotations

import ast
import json
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

from cc_agent.hooks import HookViolation, PROTECTED_NAMES, validate_test_command
from cc_agent.tools import RepoTools


TASK_TYPES = frozenset({"tool_call", "tool_strategy", "swebench_plan", "swebench_patch"})
REQUIRED_FIELDS = ("instruction", "input", "output")

_API_KEY_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key)\b\s*[:=]\s*"
        r"[\"']?(?!your-|example|dummy|<)[A-Za-z0-9_./+=-]{12,}"
    ),
)
_LOCAL_ABSOLUTE_PATHS = (
    re.compile(r"(?<![\w.-])/(?:home|Users)/[^\s'\"`]+"),
    re.compile(r"(?<![\w.-])/root(?:/[^\s'\"`]+)?"),
    re.compile(r"(?i)\b[A-Z]:\\Users\\[^\s'\"`]+"),
)
_SENSITIVE_NAMES = frozenset({".env", ".env.local", "id_rsa", "id_ed25519"})


@dataclass(frozen=True)
class ValidationIssue:
    filename: str
    line: int
    error_type: str
    reason: str
    severity: str = "error"

    def render(self) -> str:
        return f"{self.filename}:{self.line}: {self.severity}: {self.error_type}: {self.reason}"


@dataclass
class ValidationReport:
    records_checked: int = 0
    files_checked: int = 0
    inferred_task_types: Counter[str] = field(default_factory=Counter)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def issue_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(issue.error_type for issue in self.issues).items()))

    @property
    def error_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(issue.error_type for issue in self.errors).items()))

    @property
    def warning_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(issue.error_type for issue in self.warnings).items()))

    def add(
        self,
        filename: str | Path,
        line: int,
        error_type: str,
        reason: str,
        *,
        severity: str = "error",
    ) -> None:
        self.issues.append(ValidationIssue(str(filename), line, error_type, reason, severity))


@lru_cache(maxsize=1)
def registered_tool_names() -> frozenset[str]:
    """Return tool names from the actual Agent Tool Registry."""
    return frozenset(RepoTools(Path(".")).tool_names)


def infer_task_type(record: dict[str, Any], output: Any | None = None) -> str | None:
    """Infer the protocol type of a legacy sample that has no explicit task_type."""
    explicit = record.get("task_type")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    value = output if isinstance(output, dict) else record.get("output")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    if "tool" in value or "arguments" in value:
        return "tool_call"
    if "strategy" in value:
        return "tool_strategy"
    if "patch" in value:
        return "swebench_patch"
    if "plan" in value and "validation" in value:
        return "swebench_plan"
    return None


def validate_paths(
    paths: Iterable[str | Path],
    *,
    require_explicit_task_type: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    for path in _expand_paths(paths):
        report.files_checked += 1
        _validate_file(path, report, require_explicit_task_type=require_explicit_task_type)
    return report


def validate_record(
    record: Any,
    *,
    filename: str | Path = "<memory>",
    line: int = 1,
    require_explicit_task_type: bool = False,
) -> list[ValidationIssue]:
    report = ValidationReport(records_checked=1)
    _validate_record(
        record,
        filename=str(filename),
        line=line,
        report=report,
        require_explicit_task_type=require_explicit_task_type,
    )
    return report.issues


def _expand_paths(paths: Iterable[str | Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for value in paths:
        path = Path(value)
        if path.is_dir():
            json_files = (
                candidate
                for candidate in path.rglob("*.json")
                if candidate.name not in {"dataset_info.json", "dataset_stats.json"}
            )
            candidates = sorted(path.rglob("*.jsonl")) + sorted(json_files)
        else:
            candidates = [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def _validate_file(path: Path, report: ValidationReport, *, require_explicit_task_type: bool) -> None:
    if not path.exists() or not path.is_file():
        report.add(path, 1, "file_not_found", "Input path does not exist or is not a file")
        return
    if path.suffix not in {".jsonl", ".json"}:
        report.add(path, 1, "unsupported_format", "Expected a .jsonl or .json file")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        report.add(path, 1, "file_read_error", str(exc))
        return

    records = _jsonl_records(path, text, report) if path.suffix == ".jsonl" else _json_records(path, text, report)
    for line, record in records:
        report.records_checked += 1
        _validate_record(
            record,
            filename=str(path),
            line=line,
            report=report,
            require_explicit_task_type=require_explicit_task_type,
        )


def _jsonl_records(path: Path, text: str, report: ValidationReport) -> Iterator[tuple[int, Any]]:
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            yield line_number, json.loads(raw_line)
        except json.JSONDecodeError as exc:
            report.add(path, line_number, "invalid_json", f"{exc.msg} at column {exc.colno}")


def _json_records(path: Path, text: str, report: ValidationReport) -> Iterator[tuple[int, Any]]:
    decoder = json.JSONDecoder()
    index = _skip_space(text, 0)
    if index >= len(text) or text[index] != "[":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            report.add(path, exc.lineno, "invalid_json", f"{exc.msg} at column {exc.colno}")
            return
        if isinstance(value, dict):
            yield 1, value
        else:
            report.add(path, 1, "invalid_top_level", "JSON must contain one sample object or an array of sample objects")
        return

    index += 1
    while True:
        index = _skip_space(text, index)
        if index >= len(text):
            report.add(path, _line_number(text, index), "invalid_json", "Unterminated JSON array")
            return
        if text[index] == "]":
            return
        line = _line_number(text, index)
        try:
            value, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            report.add(path, exc.lineno, "invalid_json", f"{exc.msg} at column {exc.colno}")
            return
        yield line, value
        index = _skip_space(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "]":
            return
        report.add(path, _line_number(text, index), "invalid_json", "Expected ',' or ']' after array item")
        return


def _validate_record(
    record: Any,
    *,
    filename: str,
    line: int,
    report: ValidationReport,
    require_explicit_task_type: bool,
) -> None:
    if not isinstance(record, dict):
        report.add(filename, line, "invalid_record", "Each sample must be a JSON object")
        return

    missing = [name for name in REQUIRED_FIELDS if name not in record]
    if missing:
        report.add(filename, line, "missing_field", f"Missing required field(s): {', '.join(missing)}")
    for name in REQUIRED_FIELDS:
        if name in record and _is_empty(record[name]):
            report.add(filename, line, "empty_field", f"Field {name!r} must not be empty")

    instruction = record.get("instruction")
    if instruction is not None and not isinstance(instruction, str):
        report.add(filename, line, "invalid_field_type", "instruction must be a string")

    normalized_input = _decode_structured_field(record.get("input"), "input", filename, line, report)
    normalized_output = _decode_structured_field(record.get("output"), "output", filename, line, report)
    for name, normalized in (("input", normalized_input), ("output", normalized_output)):
        if name in record and not _is_empty(record[name]) and _is_empty(normalized):
            report.add(filename, line, "empty_field", f"Field {name!r} must not encode an empty object/list")

    explicit_type = record.get("task_type")
    if explicit_type is None:
        if require_explicit_task_type:
            report.add(filename, line, "missing_task_type", "Canonical SFT samples require an explicit task_type")
        task_type = infer_task_type(record, normalized_output)
        if task_type:
            report.inferred_task_types[task_type] += 1
    elif not isinstance(explicit_type, str) or explicit_type not in TASK_TYPES:
        report.add(filename, line, "invalid_task_type", f"task_type must be one of: {', '.join(sorted(TASK_TYPES))}")
        task_type = None
    else:
        task_type = explicit_type

    if task_type is None:
        report.add(filename, line, "unknown_task_type", "Could not determine the sample protocol from task_type/output")
    elif isinstance(normalized_output, dict):
        _validate_output(task_type, normalized_output, filename, line, report)

    normalized_record = {**record, "input": normalized_input, "output": normalized_output}
    if _contains_unknown_task(normalized_record):
        report.add(filename, line, "unknown_task", "Sample contains the placeholder 'Unknown task'")
    for error_type, reason, severity in _security_findings(normalized_record, task_type):
        report.add(filename, line, error_type, reason, severity=severity)

    _ = normalized_input


def _decode_structured_field(
    value: Any,
    field_name: str,
    filename: str,
    line: int,
    report: ValidationReport,
) -> Any:
    if isinstance(value, str):
        if not value.strip():
            return value
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            report.add(filename, line, "invalid_structured_field", f"Alpaca field {field_name!r} is not valid JSON: {exc.msg}")
            return None
    if value is not None and not isinstance(value, (dict, list)):
        report.add(filename, line, "invalid_field_type", f"{field_name} must be an object/list or a JSON-encoded object/list")
    return value


def _validate_output(
    task_type: str,
    output: dict[str, Any],
    filename: str,
    line: int,
    report: ValidationReport,
) -> None:
    if task_type == "tool_call":
        _validate_tool_call(output, filename, line, report)
    elif task_type == "tool_strategy":
        strategy = output.get("strategy")
        if not isinstance(strategy, list) or not strategy:
            report.add(filename, line, "invalid_strategy", "strategy must be a non-empty list")
            return
        for index, step in enumerate(strategy):
            if not isinstance(step, dict):
                report.add(filename, line, "invalid_strategy", f"strategy[{index}] must be an object")
                continue
            _validate_tool_name(step.get("tool"), filename, line, report, context=f"strategy[{index}]")
            if _is_empty(step.get("purpose")):
                report.add(filename, line, "invalid_strategy", f"strategy[{index}].purpose must not be empty")
    elif task_type == "swebench_plan":
        for name in ("plan", "validation"):
            if _is_empty(output.get(name)):
                report.add(filename, line, "invalid_plan", f"Plan output requires non-empty {name!r}")
    elif task_type == "swebench_patch":
        patch = output.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            report.add(filename, line, "invalid_patch", "Patch output requires a non-empty patch string")
        else:
            reason = validate_unified_diff(patch)
            if reason:
                report.add(filename, line, "invalid_patch", reason)


def _validate_tool_call(output: dict[str, Any], filename: str, line: int, report: ValidationReport) -> None:
    tool = output.get("tool")
    if not _validate_tool_name(tool, filename, line, report):
        return
    arguments = output.get("arguments")
    if not isinstance(arguments, dict):
        report.add(filename, line, "invalid_tool_arguments", "Tool output requires an arguments object")
        return

    required: dict[str, tuple[str, ...]] = {
        "read_file": ("path",),
        "grep": ("pattern",),
        "retrieve_context": ("query",),
        "replace_in_file": ("path", "old", "new"),
        "write_file": ("path", "content"),
    }
    for name in required.get(tool, ()):
        if name not in arguments or not isinstance(arguments[name], str) or (name not in {"old", "new", "content"} and not arguments[name].strip()):
            report.add(filename, line, "invalid_tool_arguments", f"{tool} requires string argument {name!r}")

    string_options: dict[str, tuple[str, ...]] = {
        "list_files": ("path",),
        "read_file": ("path",),
        "grep": ("path", "pattern"),
        "retrieve_context": ("query", "path", "language", "symbol"),
        "replace_in_file": ("path", "old", "new"),
        "write_file": ("path", "content"),
        "run_tests": ("command",),
        "finish": ("summary",),
    }
    for name in string_options.get(tool, ()):
        if name in arguments and not isinstance(arguments[name], str):
            report.add(filename, line, "invalid_tool_arguments", f"{tool}.{name} must be a string")

    if tool == "read_file" and "limit" in arguments and not _positive_int(arguments["limit"]):
        report.add(filename, line, "invalid_tool_arguments", "read_file.limit must be a positive integer")
    if tool == "retrieve_context" and "top_k" in arguments and not _positive_int(arguments["top_k"]):
        report.add(filename, line, "invalid_tool_arguments", "retrieve_context.top_k must be a positive integer")
    if tool == "grep" and isinstance(arguments.get("pattern"), str):
        try:
            re.compile(arguments["pattern"])
        except re.error as exc:
            report.add(filename, line, "invalid_tool_arguments", f"grep.pattern is not a valid regular expression: {exc}")
    if tool == "run_tests" and isinstance(arguments.get("command", ""), str):
        try:
            validate_test_command(arguments.get("command", ""))
        except HookViolation as exc:
            report.add(filename, line, "invalid_tool_arguments", str(exc))

    path = arguments.get("path")
    if isinstance(path, str) and _has_parent_traversal(path):
        report.add(filename, line, "invalid_tool_arguments", f"{tool}.path must not escape the repository")
    if isinstance(path, str) and _is_local_absolute_path(path):
        report.add(filename, line, "invalid_tool_arguments", f"{tool}.path must be repository-relative")
    if tool in {"write_file", "replace_in_file"} and isinstance(path, str):
        normalized_path = PurePosixPath(path.replace("\\", "/"))
        parts = normalized_path.parts
        if normalized_path.name in PROTECTED_NAMES or any(
            part in {".git", "node_modules", "__pycache__"} for part in parts
        ):
            report.add(filename, line, "invalid_tool_arguments", f"{tool}.path targets a protected or internal file")
    if tool == "write_file" and isinstance(path, str) and path.lower().endswith(".py"):
        content = arguments.get("content")
        if isinstance(content, str):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    ast.parse(content, filename=path)
            except (SyntaxError, ValueError) as exc:
                detail = f"{exc.msg} at line {exc.lineno}" if isinstance(exc, SyntaxError) else str(exc)
                report.add(filename, line, "invalid_python_syntax", f"write_file target {path!r}: {detail}")


def _validate_tool_name(
    value: Any,
    filename: str,
    line: int,
    report: ValidationReport,
    *,
    context: str = "output",
) -> bool:
    if not isinstance(value, str) or not value.strip():
        report.add(filename, line, "invalid_tool", f"{context}.tool must be a non-empty string")
        return False
    if value not in registered_tool_names():
        report.add(filename, line, "invalid_tool", f"Unknown Tool Registry name {value!r} in {context}")
        return False
    return True


def validate_unified_diff(patch: str) -> str | None:
    lines = patch.splitlines()
    starts = [index for index, value in enumerate(lines) if value.startswith("diff --git a/")]
    if not starts:
        return "Patch must contain a 'diff --git a/... b/...' header"
    starts.append(len(lines))
    for position in range(len(starts) - 1):
        section = lines[starts[position] : starts[position + 1]]
        header = section[0]
        if not re.fullmatch(r"diff --git a/\S+ b/\S+", header):
            return f"Malformed diff header: {header!r}"
        has_old = any(value.startswith("--- ") for value in section[1:])
        has_new = any(value.startswith("+++ ") for value in section[1:])
        has_hunk = any(value.startswith("@@ ") for value in section[1:])
        is_binary = any(value == "GIT binary patch" or value.startswith("Binary files ") for value in section[1:])
        if not has_old or not has_new:
            return f"Diff section {header!r} requires --- and +++ file headers"
        if not has_hunk and not is_binary:
            return f"Diff section {header!r} requires an @@ hunk header"
    return None


def _security_findings(value: dict[str, Any], task_type: str | None) -> list[tuple[str, str, str]]:
    findings: list[tuple[str, str, str]] = []
    strings = list(_walk_strings(value))
    if any(pattern.search(text) for text in strings for pattern in _API_KEY_PATTERNS):
        findings.append(("api_key_exposure", "Sample contains a credential or API-key-like value", "error"))

    output_path = _find_local_absolute_path(value.get("output"))
    metadata_path = _find_local_absolute_path(value.get("metadata"))
    other_path = _find_local_absolute_path(
        {key: item for key, item in value.items() if key not in {"input", "output", "metadata"}}
    )
    input_path = _find_local_absolute_path(value.get("input"))
    if output_path or metadata_path or other_path:
        path = output_path or metadata_path or other_path
        findings.append(("local_absolute_path", f"Sample contains local absolute path {path!r}", "error"))
    elif input_path:
        severity = "warning" if task_type in {"swebench_plan", "swebench_patch"} else "error"
        reason = f"SWE-bench issue/input contains local absolute path {input_path!r}" if severity == "warning" else f"Sample contains local absolute path {input_path!r}"
        findings.append(("local_absolute_path", reason, severity))
    sensitive = next(
        (
            path
            for path in _walk_path_values(value)
            if any(part in _SENSITIVE_NAMES for part in path.replace("\\", "/").split("/"))
        ),
        None,
    )
    if sensitive:
        findings.append(("sensitive_file", f"Sample references protected file {sensitive!r}", "error"))
    return findings


def _find_local_absolute_path(value: Any) -> str | None:
    for text in _walk_strings(value):
        for pattern in _LOCAL_ABSOLUTE_PATHS:
            if match := pattern.search(text):
                return match.group(0)
    return next((path for path in _walk_path_values(value) if _is_local_absolute_path(path)), None)


def _walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _walk_path_values(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for child_key, item in value.items():
            lowered = str(child_key).lower()
            if isinstance(item, str) and (lowered == "path" or lowered.endswith("_path") or lowered in {"file", "filename", "trace_file"}):
                yield item
            yield from _walk_path_values(item)
        patch = value.get("patch")
        if isinstance(patch, str):
            for line in patch.splitlines():
                if line.startswith(("--- ", "+++ ")):
                    yield line[4:].split("\t", 1)[0]
    elif isinstance(value, list):
        for item in value:
            yield from _walk_path_values(item)


def _contains_unknown_task(value: Any) -> bool:
    return any("unknown task" in text.lower() for text in _walk_strings(value))


def _has_parent_traversal(path: str) -> bool:
    return ".." in PurePosixPath(path.replace("\\", "/")).parts


def _is_local_absolute_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized == "/dev/null":
        return False
    return normalized.startswith("/") or bool(re.match(r"(?i)^[A-Z]:/", normalized))


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, (str, list, dict)) and not value)


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, min(index, len(text))) + 1
