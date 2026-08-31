from __future__ import annotations

import difflib
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cc_agent.hooks import HookViolation, ensure_inside_repo, post_tool_check, validate_tool_call
from cc_agent.repo_indexer import RepoIndexer


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    blocked: bool = False
    reason: str = ""


class RepoTools:
    def __init__(self, repo_path: str | Path, timeout: int = 60):
        self.repo_root = Path(repo_path).resolve()
        self.timeout = timeout
        self._tools: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "grep": self._grep,
            "retrieve_context": self._retrieve_context,
            "replace_in_file": self._replace_in_file,
            "write_file": self._write_file,
            "run_tests": self._run_tests,
            "git_diff": self._git_diff,
            "finish": self._finish,
        }

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def descriptions(self) -> str:
        return """
Available tools:
- list_files {"path": "."}: list repository files under path.
- read_file {"path": "relative/path.py"}: read a text file.
- grep {"pattern": "regex", "path": "."}: search text files with a regex.
- retrieve_context {"query": "bug or symbol name", "top_k": 5, "path": "optional/path",
  "language": "python", "symbol": "optional_symbol"}: hybrid BM25 + dense retrieval with metadata filters.
- replace_in_file {"path": "relative/path.py", "old": "exact old text", "new": "replacement text"}:
  make a focused exact replacement.
- write_file {"path": "relative/path.py", "content": "full new file content"}: overwrite a file safely.
- run_tests {"command": "pytest -q"}: run an allowlisted test command.
- git_diff {}: show git diff if available, otherwise a best-effort status.
- finish {"summary": "final answer"}: finish when the task is complete.
""".strip()

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            return ToolResult(ok=False, output=f"Unknown tool: {name}")
        try:
            validate_tool_call(self.repo_root, name, arguments)
            result = self._tools[name](arguments)
            note = post_tool_check(name, result.ok, result.output)
            if note:
                return ToolResult(result.ok, f"{result.output}\n\nHook note: {note}", result.blocked, result.reason)
            return result
        except HookViolation as exc:
            return ToolResult(ok=False, output=str(exc), blocked=True, reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - tool boundary should convert errors into observations
            return ToolResult(ok=False, output=f"Tool error: {type(exc).__name__}: {exc}")

    def _list_files(self, arguments: dict[str, Any]) -> ToolResult:
        base = ensure_inside_repo(self.repo_root, arguments.get("path", "."))
        if not base.exists():
            return ToolResult(False, f"Path does not exist: {base}")
        files: list[str] = []
        for path in sorted(base.rglob("*")) if base.is_dir() else [base]:
            if any(part in {".git", "__pycache__", "node_modules", ".pytest_cache"} for part in path.parts):
                continue
            if path.is_file():
                files.append(path.relative_to(self.repo_root).as_posix())
            if len(files) >= 120:
                files.append("... output truncated")
                break
        return ToolResult(True, "\n".join(files) or "No files found.")

    def _read_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = ensure_inside_repo(self.repo_root, arguments["path"])
        if not path.exists() or not path.is_file():
            return ToolResult(False, f"File not found: {arguments['path']}")
        text = path.read_text(encoding="utf-8", errors="replace")
        limit = int(arguments.get("limit", 12000))
        if len(text) > limit:
            text = text[:limit] + "\n... file truncated"
        return ToolResult(True, text)

    def _grep(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = arguments.get("pattern", "")
        if not pattern:
            return ToolResult(False, "grep requires a pattern")
        base = ensure_inside_repo(self.repo_root, arguments.get("path", "."))
        regex = re.compile(pattern)
        matches: list[str] = []
        targets = sorted(base.rglob("*")) if base.is_dir() else [base]
        for path in targets:
            if not path.is_file() or any(part in {".git", "node_modules", "__pycache__"} for part in path.parts):
                continue
            try:
                for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if regex.search(line):
                        rel = path.relative_to(self.repo_root).as_posix()
                        matches.append(f"{rel}:{line_no}: {line}")
                        if len(matches) >= 80:
                            return ToolResult(True, "\n".join(matches) + "\n... matches truncated")
            except OSError:
                continue
        return ToolResult(True, "\n".join(matches) or "No matches found.")

    def _retrieve_context(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query", ""))
        top_k = int(arguments.get("top_k", 5))
        if not query.strip():
            return ToolResult(False, "retrieve_context requires a non-empty query")
        output = RepoIndexer(self.repo_root).retrieve(
            query=query,
            top_k=top_k,
            path_filter=str(arguments.get("path", "")) or None,
            language_filter=str(arguments.get("language", "")) or None,
            symbol_filter=str(arguments.get("symbol", "")) or None,
        )
        return ToolResult(True, output)

    def _replace_in_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = ensure_inside_repo(self.repo_root, arguments["path"])
        old = arguments.get("old")
        new = arguments.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            return ToolResult(False, "replace_in_file requires string old and new fields")
        if not path.exists() or not path.is_file():
            return ToolResult(False, f"File not found: {arguments['path']}")
        content = path.read_text(encoding="utf-8", errors="replace")
        count = content.count(old)
        if count == 0:
            return ToolResult(False, "old text not found; inspect the file before retrying")
        if count > 1:
            return ToolResult(False, f"old text occurs {count} times; provide a more specific snippet")
        updated = content.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        rel = path.relative_to(self.repo_root).as_posix()
        diff = "\n".join(
            difflib.unified_diff(
                content.splitlines(),
                updated.splitlines(),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            )
        )
        return ToolResult(True, diff or f"Updated {rel} with no textual diff.")

    def _write_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = ensure_inside_repo(self.repo_root, arguments["path"])
        content = arguments.get("content")
        if not isinstance(content, str):
            return ToolResult(False, "write_file requires string content")
        old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        rel = path.relative_to(self.repo_root).as_posix()
        diff = "\n".join(
            difflib.unified_diff(
                old.splitlines(),
                content.splitlines(),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            )
        )
        return ToolResult(True, diff or f"Wrote {rel} with no textual diff.")

    def _run_tests(self, arguments: dict[str, Any]) -> ToolResult:
        command = arguments.get("command") or "pytest -q"
        command_parts = shlex.split(command)
        if command_parts and command_parts[0] == "pytest":
            command_parts = [sys.executable, "-m", "pytest", *command_parts[1:]]
        completed = subprocess.run(
            command_parts,
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(self.repo_root)},
            check=False,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return ToolResult(completed.returncode == 0, output or f"Command exited with {completed.returncode}")

    def _git_diff(self, arguments: dict[str, Any]) -> ToolResult:
        completed = subprocess.run(
            ["git", "diff", "--"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            return ToolResult(False, completed.stderr.strip() or "git diff failed; repository may not be a git repo")
        return ToolResult(True, completed.stdout.strip() or "No git diff.")

    def _finish(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(True, str(arguments.get("summary", "Finished.")))
