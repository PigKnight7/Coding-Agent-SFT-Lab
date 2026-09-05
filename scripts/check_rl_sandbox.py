#!/usr/bin/env python3
"""CPU-only real sandbox positive/negative controls; no model or GPU imports."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cc_agent.rl.environment import SandboxVerifier, sandbox_capability


def check_sandbox():
    backend, reason = sandbox_capability()
    print(f"[sandbox] backend={backend}; {reason}")
    with tempfile.TemporaryDirectory(prefix="rl-precheck-") as directory:
        repo = Path(directory)
        (repo / "solution.py").write_text("def add(a, b):\n    return a + b\n")
        (repo / "tests").mkdir()
        (repo / "tests/test_solution.py").write_text("from solution import add\ndef test_add():\n    assert add(1, 2) == 3\n")
        verifier = SandboxVerifier()
        if not verifier(repo, 30).success:
            raise RuntimeError("Sandbox positive control failed")
        (repo / "solution.py").write_text("def add(a, b):\n    return 0\n")
        if verifier(repo, 30).success:
            raise RuntimeError("Sandbox negative control failed")
    print(f"[PASS] Real CPU isolated pytest positive/negative controls ({backend}); no model or GPU used")
    return backend


if __name__ == "__main__":
    check_sandbox()
