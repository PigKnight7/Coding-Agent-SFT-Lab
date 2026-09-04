from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


@dataclass(frozen=True)
class BuildResult:
    tasks_path: Path
    sft_path: Path | None
    repo_dir: Path | None
    count: int

    def render(self) -> str:
        lines = [f"Built {self.count} records", f"Tasks: {self.tasks_path}"]
        if self.sft_path:
            lines.append(f"SFT: {self.sft_path}")
        if self.repo_dir:
            lines.append(f"Repos: {self.repo_dir}")
        return "\n".join(lines)


def build_mbpp(output_dir: str | Path, limit: int = 50, split: str = "test") -> BuildResult:
    dataset = _load_mbpp_rows(split=split)
    output = Path(output_dir)
    repo_dir = output / "repos" / "mbpp"
    tasks_path = output / "tasks" / "mbpp_tasks.jsonl"
    sft_path = output / "sft" / "mbpp_sft.jsonl"
    _ensure_dirs(tasks_path, sft_path, repo_dir)

    count = 0
    with tasks_path.open("w", encoding="utf-8") as tasks_file, sft_path.open("w", encoding="utf-8") as sft_file:
        for row in _take(dataset, limit):
            task_id = str(row.get("task_id", count))
            text = str(row.get("text", "")).strip()
            code = str(row.get("code", "")).strip()
            tests = row.get("test_list") or []
            if not code or not tests:
                continue

            repo_path = repo_dir / f"mbpp_{task_id}"
            skeleton = _python_skeleton_from_solution(code)
            _write_python_task_repo(repo_path, skeleton, tests)

            task_record = {
                "id": f"mbpp_{task_id}",
                "source": "MBPP",
                "repo": f"{output.name}/{repo_path.relative_to(output).as_posix()}",
                "task": f"实现 solution.py 中的函数，使其满足题意：{text}",
                "test_command": "pytest -q",
                "success_hint": "All generated pytest tests pass.",
            }
            sft_record = _make_write_file_sft(
                task=task_record["task"],
                repo_context="solution.py contains a function skeleton. tests/test_solution.py contains generated MBPP tests.",
                path="solution.py",
                content=code + "\n",
                reason="Use the reference solution to implement the requested function.",
                metadata={"id": task_record["id"], "source": "MBPP"},
            )
            tasks_file.write(json.dumps(task_record, ensure_ascii=False) + "\n")
            sft_file.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            count += 1

    return BuildResult(tasks_path=tasks_path, sft_path=sft_path, repo_dir=repo_dir, count=count)


def build_humaneval(output_dir: str | Path, limit: int = 50, split: str = "test") -> BuildResult:
    dataset = _load_humaneval_rows(split=split)
    output = Path(output_dir)
    repo_dir = output / "repos" / "humaneval"
    tasks_path = output / "tasks" / "humaneval_tasks.jsonl"
    sft_path = output / "sft" / "humaneval_sft.jsonl"
    _ensure_dirs(tasks_path, sft_path, repo_dir)

    count = 0
    with tasks_path.open("w", encoding="utf-8") as tasks_file, sft_path.open("w", encoding="utf-8") as sft_file:
        for row in _take(dataset, limit):
            task_id = _safe_id(str(row.get("task_id", count)))
            prompt = str(row.get("prompt", ""))
            canonical_solution = str(row.get("canonical_solution", ""))
            test = str(row.get("test", ""))
            entry_point = str(row.get("entry_point", ""))
            if not prompt or not canonical_solution or not test or not entry_point:
                continue

            repo_path = repo_dir / task_id
            skeleton = prompt.rstrip() + "\n    pass\n"
            solution = prompt.rstrip() + "\n" + canonical_solution.rstrip() + "\n"
            test_content = (
                f"from solution import {entry_point}\n\n"
                f"{test}\n\n"
                f"def test_humaneval() -> None:\n"
                f"    check({entry_point})\n"
            )
            _write_repo(repo_path, {"solution.py": skeleton, "tests/test_solution.py": test_content})

            task_record = {
                "id": task_id,
                "source": "HumanEval",
                "repo": f"{output.name}/{repo_path.relative_to(output).as_posix()}",
                "task": f"实现 solution.py 中的 {entry_point} 函数，使其通过 HumanEval 测试。",
                "test_command": "pytest -q",
                "success_hint": "HumanEval check passes.",
            }
            sft_record = _make_write_file_sft(
                task=task_record["task"],
                repo_context="solution.py contains the HumanEval prompt and an incomplete implementation.",
                path="solution.py",
                content=solution,
                reason="Complete the function according to the prompt and tests.",
                metadata={"id": task_record["id"], "source": "HumanEval"},
            )
            tasks_file.write(json.dumps(task_record, ensure_ascii=False) + "\n")
            sft_file.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            count += 1

    return BuildResult(tasks_path=tasks_path, sft_path=sft_path, repo_dir=repo_dir, count=count)


def build_swebench_lite(output_dir: str | Path, limit: int = 50, split: str = "test") -> BuildResult:
    dataset = _load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    output = Path(output_dir)
    tasks_path = output / "tasks" / "swebench_lite_tasks.jsonl"
    _ensure_dirs(tasks_path)

    count = 0
    with tasks_path.open("w", encoding="utf-8") as tasks_file:
        for row in _take(dataset, limit):
            instance_id = str(row.get("instance_id", count))
            task_record = {
                "id": instance_id,
                "source": "SWE-bench_Lite",
                "repo_name": row.get("repo"),
                "base_commit": row.get("base_commit"),
                "task": row.get("problem_statement"),
                "test_command": "Use SWE-bench harness for this instance.",
                "patch": row.get("patch"),
                "test_patch": row.get("test_patch"),
                "note": "This manifest does not create local repos. Use SWE-bench harness to materialize each instance.",
            }
            tasks_file.write(json.dumps(task_record, ensure_ascii=False) + "\n")
            count += 1

    return BuildResult(tasks_path=tasks_path, sft_path=None, repo_dir=None, count=count)


def swebench_to_sft(input_path: str | Path, output_path: str | Path, mode: str = "patch") -> int:
    """Convert SWE-bench Lite manifest records into SFT samples.

    mode="patch" trains direct patch generation; mode="plan" trains repair planning.
    This does not clone repositories or run tests.
    """
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as out:
        for row in _read_jsonl(source):
            if row.get("source") != "SWE-bench_Lite":
                continue
            patch = row.get("patch") or ""
            if mode == "patch" and not patch.strip():
                continue
            if mode == "plan":
                sample = {
                    "task_type": "swebench_plan",
                    "instruction": "根据真实 GitHub issue 描述，制定代码仓库修复计划。",
                    "input": {
                        "repo_name": row.get("repo_name"),
                        "base_commit": row.get("base_commit"),
                        "problem_statement": row.get("task"),
                    },
                    "output": {
                        "plan": _patch_to_plan_hint(patch),
                        "validation": "Use SWE-bench harness and the provided test_patch to validate the fix.",
                    },
                    "metadata": {"id": row.get("id"), "source": "SWE-bench_Lite"},
                }
            else:
                sample = {
                    "task_type": "swebench_patch",
                    "instruction": "根据真实 GitHub issue 描述生成 unified diff 修复补丁。",
                    "input": {
                        "repo_name": row.get("repo_name"),
                        "base_commit": row.get("base_commit"),
                        "problem_statement": row.get("task"),
                    },
                    "output": {
                        "patch": patch,
                    },
                    "metadata": {
                        "id": row.get("id"),
                        "source": "SWE-bench_Lite",
                        "test_patch_available": bool(row.get("test_patch")),
                    },
                }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def local_tasks_to_sft(input_path: str | Path, output_path: str | Path) -> int:
    """Convert local runnable task manifests into high-level SFT task samples.

    These samples teach the model the expected tool-use strategy, not the final code.
    Use traces_to_sft for actual executed tool trajectories.
    """
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as out:
        for row in _read_jsonl(source):
            if not row.get("repo") or not row.get("task"):
                continue
            sample = {
                "task_type": "tool_strategy",
                "instruction": "根据本地代码任务，制定并执行最小工具调用策略。",
                "input": {
                    "repo": row.get("repo"),
                    "task": row.get("task"),
                    "test_command": row.get("test_command", "pytest -q"),
                },
                "output": {
                    "strategy": [
                        {"tool": "retrieve_context", "purpose": "定位相关代码和测试"},
                        {"tool": "read_file", "purpose": "读取需要修改的文件"},
                        {"tool": "replace_in_file", "purpose": "进行最小安全修改"},
                        {"tool": "run_tests", "purpose": "验证修改"},
                    ]
                },
                "metadata": {"id": row.get("id"), "source": row.get("source")},
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def traces_to_sft(trace_path: str | Path, output_path: str | Path) -> int:
    traces = _trace_files(trace_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as out:
        for trace_file in traces:
            task = ""
            plan = ""
            history: list[dict[str, Any]] = []
            for record in _read_jsonl(trace_file):
                event = record.get("event")
                payload = record.get("payload", {})
                if event == "repo_indexed":
                    task = payload.get("task", task)
                elif event == "plan":
                    plan = payload.get("plan", "")
                elif event == "tool_call":
                    action = payload.get("action", {})
                    result = payload.get("result", {})
                    if action.get("tool") and action.get("tool") != "finish" and result.get("ok"):
                        resolved_task = task or recover_task_from_plan(plan)
                        if not resolved_task:
                            history.append({"action": action, "result": result})
                            continue
                        sample = {
                            "task_type": "tool_call",
                            "instruction": "根据用户任务、计划和已有工具轨迹，选择下一步工具调用。",
                            "input": {
                                "task": resolved_task,
                                "plan": plan,
                                "history": history[-6:],
                            },
                            "output": {
                                "tool": action.get("tool"),
                                "arguments": action.get("arguments", {}),
                                "reason": action.get("reason", ""),
                            },
                            "metadata": {
                                "trace_file": str(trace_file),
                                "step": payload.get("step"),
                            },
                        }
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        count += 1
                    history.append({"action": action, "result": result})
    return count


def _load_mbpp_rows(split: str) -> list[dict[str, Any]]:
    _ensure_hf_endpoint()
    try:
        from datasets import load_dataset

        return list(load_dataset("google-research-datasets/mbpp", split=split))
    except ImportError:
        pass

    import requests

    url = f"{_hf_endpoint()}/datasets/google-research-datasets/mbpp/resolve/main/sanitized-mbpp.json"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise RuntimeError("Unexpected MBPP JSON format")
        return rows
    except requests.RequestException:
        return _offline_mbpp_rows()


def _load_humaneval_rows(split: str) -> list[dict[str, Any]]:
    _ensure_hf_endpoint()
    try:
        from datasets import load_dataset

        return list(load_dataset("openai/openai_humaneval", split=split))
    except ImportError:
        pass

    import requests

    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        content = gzip.decompress(response.content).decode("utf-8")
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    except requests.RequestException:
        return _offline_humaneval_rows()


def _load_dataset(name: str, split: str):
    _ensure_hf_endpoint()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            f"The 'datasets' package is required for {name}. Install with conda if pip fails: "
            "python -m pip install datasets pyarrow"
        ) from exc
    try:
        return load_dataset(name, split=split)
    except Exception as exc:
        endpoint = _hf_endpoint()
        raise RuntimeError(
            f"Failed to load {name!r} from HuggingFace endpoint {endpoint!r}. "
            "If you are in China, try: export HF_ENDPOINT=https://hf-mirror.com ; "
            "if still failing, download/cache the dataset manually."
        ) from exc


def _ensure_hf_endpoint() -> None:
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)


def _hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")


def _offline_mbpp_rows() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "offline_001",
            "text": "Write a function to subtract two numbers.",
            "code": "def subtract(a, b):\n    return a - b",
            "test_list": ["assert subtract(5, 3) == 2", "assert subtract(-1, -3) == 2"],
        },
        {
            "task_id": "offline_002",
            "text": "Write a function to return the square of a number.",
            "code": "def square(n):\n    return n * n",
            "test_list": ["assert square(4) == 16", "assert square(-3) == 9"],
        },
    ]


def _offline_humaneval_rows() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "HumanEval/offline_001",
            "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Return a plus b.\"\"\"\n",
            "canonical_solution": "    return a + b\n",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
            "entry_point": "add",
        }
    ]


def _take(dataset: Iterable[dict[str, Any]], limit: int) -> Iterable[dict[str, Any]]:
    for index, item in enumerate(dataset):
        if index >= limit:
            break
        yield item


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _write_python_task_repo(repo_path: Path, skeleton: str, tests: list[str]) -> None:
    test_lines = "from solution import *\n\n\ndef test_mbpp_generated() -> None:\n"
    for test in tests:
        test_lines += f"    {test}\n"
    _write_repo(repo_path, {"solution.py": skeleton, "tests/test_solution.py": test_lines})


def _write_repo(repo_path: Path, files: dict[str, str]) -> None:
    for rel_path, content in files.items():
        target = repo_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (repo_path / "AGENT.md").write_text(
        "# Generated Coding Task Repo\n\n- Make minimal edits.\n- Run `pytest -q` after edits.\n",
        encoding="utf-8",
    )


def _python_skeleton_from_solution(code: str) -> str:
    match = re.search(r"^def\s+\w+\s*\([^\n]*\)\s*(?:->\s*[^:]+)?:", code, flags=re.MULTILINE)
    if not match:
        return "# TODO: implement solution\n"
    signature = match.group(0)
    return f"{signature}\n    pass\n"


def _make_write_file_sft(
    task: str,
    repo_context: str,
    path: str,
    content: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "task_type": "tool_call",
        "instruction": "根据用户任务和仓库上下文，选择下一步工具调用。",
        "input": {
            "task": task,
            "repo_context": repo_context,
            "history": [],
        },
        "output": {
            "tool": "write_file",
            "arguments": {
                "path": path,
                "content": content,
            },
            "reason": reason,
        },
    }
    if metadata:
        record["metadata"] = metadata
    return record


def recover_task_from_plan(plan: str) -> str | None:
    """Conservatively recover a missing legacy trace task from its recorded plan."""
    if not plan.strip():
        return None
    # Keep the heuristic intentionally narrow: require both a Python file and a
    # backticked identifier that is mentioned as a function in the trace plan.
    python_files = re.findall(r"`([^`\n]+\.py)`", plan)
    identifiers = re.findall(r"`([A-Za-z_]\w*)`", plan)
    ignored = {"pytest", "python", "true", "false", "pass"}
    symbols = [value for value in identifiers if value.lower() not in ignored]
    if not python_files or not symbols:
        return None
    return f"检查并修复 {python_files[0]} 中的 {symbols[0]} 函数，并运行测试验证。"


def _patch_to_plan_hint(patch: str) -> str:
    files = re.findall(r"^diff --git a/(.*?) b/", patch, flags=re.MULTILINE)
    if not files:
        return "Inspect the issue, locate relevant files, make a minimal fix, then run the SWE-bench tests."
    unique_files = []
    for file in files:
        if file not in unique_files:
            unique_files.append(file)
    return (
        "Inspect and modify these likely relevant files: "
        + ", ".join(unique_files[:8])
        + ". Make the smallest behavior-preserving fix and validate with the provided tests."
    )


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def _trace_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_dir():
        return sorted(root.glob("*.jsonl"))
    return [root]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
