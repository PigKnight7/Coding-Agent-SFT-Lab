#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


def latest_adapter_path(path: Path) -> Path:
    if (path / "adapter_config.json").is_file():
        return path
    checkpoints = []
    for candidate in path.glob("checkpoint-*"):
        try:
            step = int(candidate.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if (candidate / "adapter_config.json").is_file():
            checkpoints.append((step, candidate))
    if not checkpoints:
        raise FileNotFoundError(f"No loadable LoRA adapter found under {path}")
    return max(checkpoints)[1]


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    return True


def verify_training_artifacts(path: Path, *, expected_steps: int = 2) -> tuple[Path, dict[str, Any]]:
    """Apply the non-model Smoke gates: completed steps plus a saved adapter."""
    adapter_path = latest_adapter_path(path)
    if not any((adapter_path / name).is_file() for name in ADAPTER_WEIGHT_NAMES):
        raise FileNotFoundError(f"Adapter weights are missing under {adapter_path}")
    state_candidates = (path / "trainer_state.json", adapter_path / "trainer_state.json")
    state_path = next((candidate for candidate in state_candidates if candidate.is_file()), None)
    if state_path is None:
        raise FileNotFoundError("trainer_state.json is missing; Smoke completion cannot be proven")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    step = state.get("global_step")
    if not isinstance(step, int) or step < expected_steps:
        raise ValueError(f"Smoke is incomplete: global_step={step!r}, expected at least {expected_steps}")
    if not _all_numbers_finite(state):
        raise ValueError("trainer_state.json contains NaN or Inf")
    return adapter_path, state


def load_sample(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise ValueError(f"{path} must contain a non-empty sample array")
    return rows[0]


def is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except (json.JSONDecodeError, TypeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply hard gates to a completed Smoke LoRA run.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/qwen3_5_2b_smoke_lora"))
    parser.add_argument("--sample", type=Path, default=Path("data/llamafactory/smoke_alpaca.json"))
    parser.add_argument("--expected-steps", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for adapter verification")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by the selected GPU/PyTorch build")
    adapter_path, state = verify_training_artifacts(args.adapter_dir, expected_steps=args.expected_steps)
    sample = load_sample(args.sample)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    base_model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.eval()

    adapter_parameters = [parameter for name, parameter in model.named_parameters() if "lora_" in name]
    if not adapter_parameters:
        raise RuntimeError("Adapter reloaded but no LoRA parameters were found")
    if any(not torch.isfinite(parameter).all().item() for parameter in adapter_parameters):
        raise RuntimeError("Adapter contains NaN or Inf")

    messages = [
        {"role": "system", "content": sample.get("system", "")},
        {"role": "user", "content": f"{sample['instruction']}\n\n{sample.get('input', '')}"},
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=[prompt], return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    if any(not torch.isfinite(score).all().item() for score in generated.scores):
        raise RuntimeError("Generation scores contain NaN or Inf")
    new_tokens = generated.sequences[0, inputs["input_ids"].shape[1] :]
    text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("Adapter reloaded, but minimal generation returned an empty response")

    print(f"training_complete=true global_step={state['global_step']}")
    print(f"adapter_saved=true adapter={adapter_path}")
    print("adapter_reloaded=true")
    print(f"generated_nonempty=true generated_tokens={new_tokens.numel()}")
    print("finite=true")
    print(f"diagnostic_json_valid={str(is_json_object(text)).lower()}")
    print(text)


if __name__ == "__main__":
    main()
