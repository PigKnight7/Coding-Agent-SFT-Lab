from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Termination(str, Enum):
    FINISH = "finish"
    MAX_TURNS = "max_turns"
    TOOL_TIMEOUT = "tool_timeout"
    TOKEN_LIMIT = "token_limit"
    FAILED = "task_failed"
    PROTECTED = "protected_file_changed"


@dataclass(frozen=True)
class Task:
    task_id: str
    group_id: str
    split: str
    repo: str
    instruction: str
    editable: tuple[str, ...] = ("solution.py",)
    test_command: str = "pytest -q"
    hidden_tests: str | None = None


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Observation:
    ok: bool
    output: str
    blocked: bool = False
    reason: str = ""
    legal: bool = True
    edited: bool = False
    repeated: bool = False


@dataclass(frozen=True)
class Verification:
    passed: int = 0
    total: int = 0
    intact: bool = True
    timeout: bool = False
    independent: bool = False
    hidden_total: int = 0

    @property
    def success(self):
        return self.intact and not self.timeout and self.total > 0 and self.passed == self.total


@dataclass
class Trajectory:
    task_id: str
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    termination: Termination | None = None
    verification: Verification = field(default_factory=Verification)
    tested_current_edit: bool = False
    changed: bool = False
    format_errors: int = 0

    @property
    def model_tokens(self):
        return sum(self.loss_mask)

    def append_tokens(self, ids, *, model):
        if self.termination is not None:
            raise ValueError("Trajectory already terminated")
        self.completion_ids.extend(ids)
        self.loss_mask.extend([int(model)] * len(ids))

    def close(self, reason):
        if self.termination is not None:
            raise ValueError("Duplicate termination")
        self.termination = reason


@dataclass(frozen=True)
class Reward:
    components: dict[str, float]

    @property
    def total(self):
        return sum(self.components.values())
