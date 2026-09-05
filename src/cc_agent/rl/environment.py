"""Host tools never execute candidate code; verification is isolated and fail-closed."""
from __future__ import annotations

import ast
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from cc_agent.hooks import HookViolation, ensure_inside_repo, validate_tool_call
from cc_agent.tools import RepoTools
from cc_agent.rl.protocol import Observation, Verification
from cc_agent.rl.hidden import partition_tests

SCHEMAS = {
    "list_files": ({}, {"path": str}),
    "read_file": ({"path": str}, {}),
    "grep": ({"pattern": str}, {"path": str}),
    "retrieve_context": ({"query": str}, {"path": str, "top_k": int}),
    "write_file": ({"path": str, "content": str}, {}),
    "replace_in_file": ({"path": str, "old": str, "new": str}, {}),
    "run_tests": ({}, {"command": str}),
    "git_diff": ({}, {}),
    "finish": ({"summary": str}, {}),
}
SAFE_MODULES = {"math", "re", "collections", "itertools", "functools", "heapq", "bisect", "string", "typing", "statistics", "decimal", "fractions", "operator", "array"}
FORBIDDEN = {"print", "eval", "exec", "compile", "open", "input", "breakpoint", "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir", "help", "type", "object", "super", "exit", "quit", "memoryview"}
SAFE_ATTRIBUTES = set("append extend insert pop remove clear copy count index reverse sort keys values items get setdefault update fromkeys add discard union intersection difference symmetric_difference issubset issuperset isdisjoint intersection_update difference_update symmetric_difference_update lower upper strip lstrip rstrip split rsplit splitlines join replace find rfind startswith endswith isalpha isdigit isalnum isspace islower isupper title capitalize swapcase center ljust rjust zfill partition rpartition format encode decode translate maketrans expandtabs casefold removeprefix removesuffix real imag numerator denominator conjugate bit_length bit_count as_integer_ratio is_integer sqrt isqrt gcd lcm factorial comb perm ceil floor trunc fabs fsum prod exp log log2 log10 pow sin cos tan asin acos atan atan2 sinh cosh tanh radians degrees hypot dist isclose isfinite isinf isnan copysign frexp ldexp modf remainder fmod pi e tau inf nan search match fullmatch findall finditer sub subn escape group groups groupdict start end span Counter defaultdict deque OrderedDict most_common elements total appendleft popleft rotate extendleft heappush heappop heapify heappushpop heapreplace nlargest nsmallest bisect bisect_left bisect_right insort insort_left insort_right accumulate chain combinations combinations_with_replacement permutations product repeat starmap takewhile dropwhile zip_longest groupby islice filterfalse cycle count reduce cache lru_cache partial mean median median_low median_high mode multimode pstdev pvariance stdev variance Decimal Fraction ascii_letters ascii_lowercase ascii_uppercase digits punctuation whitespace ascii isdecimal isnumeric translate".split())
SAFE_ATTRIBUTES.update("List Tuple Dict Set Optional Union Iterable Sequence Any Callable Generator DefaultDict Deque".split())


def validate_candidate(text):
    """Conservative benchmark subset, in addition to OS isolation, to protect test runner integrity."""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(n not in SAFE_MODULES for n in names) or getattr(node, "level", 0):
                raise ValueError("Unsupported import in candidate")
            if any(a.name.startswith("_") or (a.asname or "").startswith("_") for a in node.names):
                raise ValueError("Private import")
            if isinstance(node, ast.ImportFrom) and any(a.name not in SAFE_ATTRIBUTES for a in node.names):
                raise ValueError("Unsupported imported symbol")
        if isinstance(node, ast.Name) and (node.id.startswith("_") and node.id != "_" or node.id in FORBIDDEN):
            raise ValueError("Unsafe candidate name")
        if isinstance(node, ast.Attribute) and (node.attr not in SAFE_ATTRIBUTES or isinstance(node.ctx, (ast.Store, ast.Del))):
            raise ValueError("Unsafe attribute access")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.Delete)):
            raise ValueError("Unsupported candidate construct")
    return tree


def stub_solution(text):
    tree = ast.parse(text)
    found = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = [ast.Raise(exc=ast.Call(func=ast.Name(id="NotImplementedError", ctx=ast.Load()), args=[], keywords=[]), cause=None)]
            node.decorator_list = []
            found = True
        elif not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
            raise ValueError("Only function benchmark tasks are supported")
    if not found:
        raise ValueError("Task has no executable target functions")
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and not any(x in p.parts for x in (".git", "__pycache__", ".pytest_cache"))}


def _bwrap_prefix():
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]
    for path in ("/usr", "/lib", "/lib64", "/bin"):
        if Path(path).exists():
            cmd += ["--ro-bind", path, path]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    # Bind virtualenv AFTER /tmp tmpfs (venvs can themselves live in /tmp).
    runtime_prefix = Path(os.sys.prefix).resolve()
    if runtime_prefix != Path("/usr") and Path("/usr") not in runtime_prefix.parents:
        cmd += ["--ro-bind", str(runtime_prefix), str(runtime_prefix)]
    return cmd


@lru_cache(maxsize=1)
def sandbox_capability():
    if not shutil.which("bwrap"):
        return "lightweight", "bubblewrap is not installed"
    try:
        probe = subprocess.run(_bwrap_prefix() + ["--", os.sys.executable, "-I", "-c", "import pytest"],
                               capture_output=True, text=True, timeout=10)
        if probe.returncode == 0:
            return "bubblewrap", "namespace and isolated Python/pytest probe passed"
        return "lightweight", probe.stderr.strip()[:500]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "lightweight", str(exc)[:500]


def sandbox_command(repo, report, *, backend="bubblewrap"):
    # Neither argv nor environment is supplied by the actor.
    work = "/work" if backend == "bubblewrap" else str(repo)
    output = "/report/result.xml" if backend == "bubblewrap" else str(report / "result.xml")
    pytest = [os.sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
              "-c", "/dev/null", "--confcutdir=" + work, "--junitxml=" + output, "tests"]
    if backend == "lightweight":
        return pytest
    if not shutil.which("bwrap"):
        raise RuntimeError("bubblewrap unavailable")
    return _bwrap_prefix() + [
        "--ro-bind", str(repo), "/work", "--bind", str(report), "/report",
        "--chdir", "/work", "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
        "--setenv", "PYTHONPATH", "/work", "--setenv", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1",
        "--setenv", "PYTHONNOUSERSITE", "1", "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--", *pytest]


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


class SandboxVerifier:
    def __init__(self, *, backend=None):
        self.backend = backend or sandbox_capability()[0]
        if self.backend not in {"bubblewrap", "lightweight"}:
            raise ValueError("Unknown sandbox backend")

    def __call__(self, repo, timeout):
        repo = Path(repo)
        try:
            validate_candidate((repo / "solution.py").read_text())
        except SyntaxError:
            return Verification(0, 1)
        except ValueError:
            return Verification(0, 1, intact=False)
        with tempfile.TemporaryDirectory(prefix="rl-verifier-") as directory:
            private = Path(directory)
            report = private / "report"
            report.mkdir()
            work = private / "repo"
            shutil.copytree(repo, work)
            before = hashes(work)
            environment = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(work),
                           "PYTHONNOUSERSITE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                           "PYTHONDONTWRITEBYTECODE": "1"}
            with (report / "stderr.log").open("wb") as errors:
                proc = subprocess.Popen(sandbox_command(work, report, backend=self.backend),
                                        cwd=work, env=environment, stdout=subprocess.DEVNULL,
                                        stderr=errors, start_new_session=True, preexec_fn=_limits)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return Verification(timeout=True)
            if hashes(work) != before or any(p.is_symlink() for p in work.rglob("*")):
                return Verification(intact=False)
            result = report / "result.xml"
            if not result.exists():
                if code in (-signal.SIGKILL, -signal.SIGXCPU):
                    return Verification(total=1, timeout=True)
                detail = (report / "stderr.log").read_text(errors="replace")[:2000]
                raise RuntimeError(f"Isolated verifier failed to produce a report (exit={code}): {detail}")
            cases = ET.parse(result).findall(".//testcase")
            passed = sum(not any(c.find(tag) is not None for tag in ("failure", "error", "skipped")) for c in cases)
            total = max(1, len(cases))
            if code != 0 and passed == total:
                total += 1
            return Verification(passed, total)


def _tool_worker(root, timeout, name, args, connection, limit):
    try:
        _limits()
        os.chdir(root)
        os.environ.clear()
        os.environ.update({"PATH": "/usr/bin:/bin", "CC_AGENT_RETRIEVAL_MODE": "lexical",
                           "RAG_INDEX_DIR": str(Path(root).parent / "rag-cache"), "PYTHON_DOTENV_DISABLED": "1"})
        result = RepoTools(root, timeout).run(name, args)
        connection.send((result.ok, result.output[:limit], result.blocked, result.reason))
    finally:
        connection.close()


class Environment:
    def __init__(self, task, *, timeout=30, output_limit=4000, verifier=None):
        self.task, self.timeout, self.output_limit = task, timeout, output_limit
        self.verifier = verifier or SandboxVerifier()
        self.temp = None

    def __enter__(self):
        if self.task.test_command != "pytest -q" or self.task.editable != ("solution.py",):
            raise ValueError("First version requires solution.py and fixed pytest verification")
        source = Path(self.task.repo).resolve()
        if any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError("Symlinks are not allowed in task repositories")
        self.temp = tempfile.TemporaryDirectory(prefix="agentic-rl-")
        try:
            self.root = Path(self.temp.name) / "repo"
            self.root.mkdir()
            (self.root / "tests").mkdir()
            if self.task.hidden_tests:
                self.partition = None
                shutil.copytree(source / "tests", self.root / "tests", dirs_exist_ok=True)
            else:
                self.partition = partition_tests(source)
                (self.root / "tests/test_solution.py").write_text(self.partition.public)
            # Project only interface stub and public tests: original README/metadata can contain answers.
            (self.root / "solution.py").write_text(stub_solution((source / "solution.py").read_text()))
            self.hard_failure = False
            self.tools = RepoTools(self.root, self.timeout)
            self.initial = hashes(self.root)
            self.seen = set()
            self.tested_hash = None
            self.last_verification = Verification()
            return self
        except BaseException:
            self.temp.cleanup()
            raise

    def __exit__(self, *args):
        self.temp.cleanup()

    def intact(self):
        if any(p.is_symlink() for p in self.root.rglob("*")):
            return False
        current = hashes(self.root)
        return {k: v for k, v in current.items() if k not in self.task.editable} == {
            k: v for k, v in self.initial.items() if k not in self.task.editable}

    def edit_hash(self):
        return tuple(hashes(self.root).get(p) for p in self.task.editable)

    @property
    def changed(self):
        return self.edit_hash() != tuple(self.initial[p] for p in self.task.editable)

    def verify(self):
        """Terminal-only private validation; never called by an actor tool."""
        if self.hard_failure or not self.intact():
            return Verification(intact=False)
        with tempfile.TemporaryDirectory(prefix="rl-hidden-verification-") as directory:
            private_repo = Path(directory) / "repo"
            shutil.copytree(self.root, private_repo)
            if self.partition:
                (private_repo / "tests/test_hidden.py").write_text(self.partition.hidden)
                hidden_count = self.partition.hidden_count
            else:
                hidden = Path(self.task.hidden_tests).resolve()
                if self.root in hidden.parents or not hidden.is_dir() or any(p.is_symlink() for p in hidden.rglob("*")):
                    raise ValueError("Hidden tests must be a trusted external directory without symlinks")
                shutil.copytree(hidden, private_repo / "tests/hidden")
                hidden_count = sum(isinstance(n, ast.Assert) for p in hidden.rglob("test_*.py") for n in ast.walk(ast.parse(p.read_text())))
            result = self.verifier(private_repo, self.timeout)
        if not self.intact():
            return Verification(intact=False)
        return replace(result, independent=True, hidden_total=hidden_count)

    def run(self, name, arguments):
        if self.hard_failure or not self.intact():
            self.hard_failure = True
            return Observation(False, "Episode terminated by integrity guard", True, "protected", False)
        key = json.dumps([name, arguments], sort_keys=True)
        repeated = key in self.seen
        self.seen.add(key)
        try:
            if name not in SCHEMAS:
                if "command" in arguments or name in {"exec", "shell", "delete_file"}:
                    raise HookViolation("Arbitrary commands and deletion are forbidden")
                raise ValueError("Unknown tool")
            required, optional = SCHEMAS[name]
            if name == "run_tests" and set(arguments) - {"command"}:
                raise HookViolation("Actor may not supply test output, status, or verifier settings")
            if set(arguments) - (required.keys() | optional.keys()) or not required.keys() <= arguments.keys():
                raise ValueError("Invalid argument keys")
            if any(type(v) is not (required | optional)[k] for k, v in arguments.items()):
                raise ValueError("Invalid argument types")
            if "path" in arguments:
                path = ensure_inside_repo(self.root, arguments["path"])
                if name in {"write_file", "replace_in_file"} and path.relative_to(self.root).as_posix() not in self.task.editable:
                    raise HookViolation("Only target solution files are editable; tests and configuration are protected")
            if name == "run_tests" and arguments.get("command", self.task.test_command) != self.task.test_command:
                raise HookViolation("Verification command is immutable")
            validate_tool_call(self.root, name, arguments)
        except (ValueError, HookViolation) as exc:
            self.hard_failure = isinstance(exc, HookViolation)
            return Observation(False, str(exc), isinstance(exc, HookViolation), "invalid_tool", False, repeated=repeated)
        if name == "run_tests":
            self.last_verification = self.verifier(self.root, self.timeout)
            self.tested_hash = self.edit_hash()
            v = self.last_verification
            self.hard_failure = not v.intact
            return Observation(v.success, f"Public tests: {v.passed}/{v.total}", blocked=not v.intact,
                               legal=v.intact, reason="timeout" if v.timeout else "", repeated=repeated)
        before = self.edit_hash()
        if name == "finish":
            return Observation(True, arguments["summary"][:self.output_limit], repeated=repeated)
        # Bound regex/retrieval/read operations too; candidate code is never imported here.
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_tool_worker, args=(self.root, self.timeout, name, arguments, sender, self.output_limit))
        process.start()
        sender.close()
        try:
            if not receiver.poll(self.timeout):
                process.kill()
                return Observation(False, "Tool timed out", reason="timeout", repeated=repeated)
            try:
                ok, output, blocked, reason = receiver.recv()
            except EOFError:
                return Observation(False, "Tool worker failed", reason="worker_failed", repeated=repeated)
        finally:
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            receiver.close()
        edited = before != self.edit_hash()
        if name in {"write_file", "replace_in_file"} and edited:
            try:
                validate_candidate((self.root / "solution.py").read_text())
            except SyntaxError:
                pass  # Ordinary syntax mistakes are task failures, not integrity violations.
            except ValueError:
                self.hard_failure = True
                return Observation(False, "Candidate violates restricted Python policy", True, "protected", False, edited)
        if blocked:
            self.hard_failure = True
        if name in {"write_file", "replace_in_file"} and not edited:
            reason = "empty_edit"
        return Observation(ok, output, blocked, reason, not blocked, edited, repeated)
