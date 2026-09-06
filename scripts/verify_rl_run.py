#!/usr/bin/env python3
"""Verify saved CPU-readable evidence from a real GRPO run; does not claim GPU validation itself."""
import argparse
import json
import math
from pathlib import Path


def verify_run(path, expected_steps):
    path = Path(path)
    state = json.loads((path / "trainer_state.json").read_text())
    if state.get("global_step", 0) < expected_steps:
        raise ValueError("GRPO did not complete the required steps")
    losses = [row["loss"] for row in state.get("log_history", []) if "loss" in row]
    if not losses or any(not math.isfinite(x) for x in losses):
        raise ValueError("Missing or non-finite real training loss")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (path / "final_adapter" / name).is_file():
            raise ValueError("Final adapter missing")
    audit = json.loads((path / "model_audit.json").read_text())
    if not audit.get("weights_changed") or not any(s["nonzero"] for s in audit.get("gradient_steps", [])):
        raise ValueError("No proven LoRA weight update/nonzero gradient")
    expected = {"loss_type": "dapo", "epsilon": 0.2, "epsilon_high": 0.28,
                "mask_truncated_completions": True, "num_iterations": 2, "use_vllm": False}
    if any(audit["grpo_kwargs"].get(k) != v for k, v in expected.items()):
        raise ValueError("Trainer DAPO settings differ from audited settings")
    rows = [json.loads(line) for line in (path / "rollouts.jsonl").read_text().splitlines()]
    # A candidate is evidence only after the entire callback batch was selected.
    selected_batches = {r["payload"]["batch_id"] for r in rows if r["event"] == "rl_batch_selected"}
    accepted = {rid for r in rows if r["event"] == "rl_group_attempt"
                and r["payload"]["selection"] == "accepted"
                and r["payload"]["batch_id"] in selected_batches
                for rid in r["payload"]["rollout_ids"]}
    trajectories = [r["payload"] for r in rows if r["event"] == "rl_trajectory"
                    and r["payload"].get("rollout_id") in accepted]
    if not any(len(t["actions"]) >= 2 and any(o["legal"] for o in t["observations"]) for t in trajectories):
        raise ValueError("No real multi-turn legal-tool trajectory; Smoke gate not satisfied")
    if any(len(r["payload"]["loss_mask"]) != len(r["payload"]["completion_ids"])
           for r in rows if r["event"] == "rl_trajectory"):
        raise ValueError("Malformed token mask")
    return {"steps": state["global_step"], "trajectories": len(trajectories), "finite_loss": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run_dir, args.expected_steps)))


if __name__ == "__main__":
    main()
