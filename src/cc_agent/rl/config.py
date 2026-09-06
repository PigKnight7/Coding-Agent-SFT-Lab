from dataclasses import dataclass, fields
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class RLConfig:
    loss_type: str = "dapo"
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    mask_truncated_completions: bool = True
    beta: float = 0.0
    num_iterations: int = 2
    num_generations: int = 4
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_turns: int = 8
    max_model_tokens: int = 2048
    max_action_tokens: int = 512
    max_context_tokens: int = 8192
    safe_length: int = 1536
    length_cap: float = 1.0
    tool_timeout: int = 30
    output_limit: int = 4000
    learning_rate: float = 5e-6
    max_steps: int = 200
    save_steps: int = 25
    seed: int = 42
    temperature: float = 1.0
    dynamic_sampling_enabled: bool = True
    dynamic_sampling_max_retries: int = 3
    zero_variance_epsilon: float = 1e-6

    def validate(self):
        for k, v in {"loss_type": "dapo", "epsilon": 0.2, "epsilon_high": 0.28, "mask_truncated_completions": True,
                     "num_iterations": 2, "beta": 0.0, "bf16": True, "gradient_checkpointing": True}.items():
            if getattr(self, k) != v:
                raise ValueError(f"{k} must be {v}")
        if type(self.dynamic_sampling_enabled) is not bool:
            raise ValueError("dynamic_sampling_enabled must be boolean")
        if type(self.dynamic_sampling_max_retries) is not int or not 0 <= self.dynamic_sampling_max_retries <= 16:
            raise ValueError("dynamic_sampling_max_retries must be an integer in [0, 16]")
        if type(self.zero_variance_epsilon) not in (int, float) or not math.isfinite(self.zero_variance_epsilon) or not 0 <= self.zero_variance_epsilon <= 1e-4:
            raise ValueError("zero_variance_epsilon must be finite in [0, 1e-4]")
        if self.num_generations not in (2, 4):
            raise ValueError("num_generations must be 2 or 4")
        for k in ("max_turns", "max_model_tokens", "max_action_tokens", "max_context_tokens", "tool_timeout", "output_limit", "max_steps", "save_steps"):
            if type(getattr(self, k)) is not int or getattr(self, k) <= 0:
                raise ValueError(f"{k} must be a positive integer")
        if not 0 <= self.safe_length < self.max_model_tokens or not 0 <= self.length_cap <= 2:
            raise ValueError("Invalid soft length penalty")
        if self.max_context_tokens <= self.max_model_tokens or self.temperature != 1.0 or self.learning_rate <= 0:
            raise ValueError("Invalid context, sampling, or learning rate")
        return self


def load_config(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(RLConfig)}:
        raise ValueError("Unknown RL configuration keys")
    return RLConfig(**data).validate()
