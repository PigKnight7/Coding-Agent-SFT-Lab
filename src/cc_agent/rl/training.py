"""Public TRL 1.12.0 integration. No Trainer overrides or patched loss implementations."""
from dataclasses import asdict
import importlib.metadata
import inspect
import json
import math
import struct
import uuid
from pathlib import Path

from cc_agent.rl.rollout import Generation, rollout, task_prompt
from cc_agent.rl.rewards import primary_outcome, optimization_rewards
from cc_agent.tracing import append_trace

TRL_VERSION = "1.12.0"


def check_trl_api():
    from trl import GRPOConfig, GRPOTrainer
    for name, version in {"trl": TRL_VERSION, "transformers": "5.5.0", "peft": "0.18.1"}.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Requires {name}=={version}")
    if "rollout_func" not in inspect.signature(GRPOTrainer).parameters:
        raise RuntimeError("TRL lacks public rollout_func")
    # env_mask is an experimental return contract: fail closed if the pinned source changes.
    source = inspect.getsource(GRPOTrainer)
    for contract in ('extra_fields.pop("env_mask", None)', 'self.rollout_func(prompts, self)',
                     'self.args.steps_per_generation * self.num_iterations',
                     'completion_mask * inputs["tool_mask"]', 'old_per_token_logps'):
        if contract not in source:
            raise RuntimeError(f"TRL rollout/mask/reuse contract changed: {contract}")
    for key in ("loss_type", "epsilon", "epsilon_high", "mask_truncated_completions", "num_iterations"):
        if key not in GRPOConfig.__dataclass_fields__:
            raise RuntimeError(f"Missing TRL capability: {key}")


class ModelPolicy:
    def __init__(self, model, tokenizer, *, sample=True):
        self.model, self.tokenizer, self.sample = model, tokenizer, sample

    def generate(self, ids, budget):
        import torch
        inputs = torch.tensor([ids], device=next(self.model.parameters()).device)
        with torch.no_grad():
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                         max_new_tokens=budget, do_sample=self.sample,
                                         temperature=1.0, top_p=1.0, top_k=0,
                                         repetition_penalty=1.0,
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         eos_token_id=self.tokenizer.eos_token_id, use_cache=False)
        generated = output[0, len(ids):].tolist()
        truncated = len(generated) >= budget and generated[-1] != self.tokenizer.eos_token_id
        return Generation(generated, self.tokenizer.decode(generated, skip_special_tokens=True), truncated)


class RolloutBridge:
    def __init__(self, tasks, tokenizer, config, trace_path, *, verifier=None, policy_factory=None):
        self.tasks = {t.task_id: t for t in tasks}
        self.tokenizer, self.config, self.trace_path = tokenizer, config.validate(), trace_path
        self.last_statistics = {}
        self.verifier, self.policy_factory = verifier, policy_factory

    def __call__(self, prompts, trainer):
        # TRL's RepeatSampler already repeats dataset rows G times. Return one rollout per input row.
        task_ids = [json.loads(p[-1]["content"])["task_id"] for p in prompts]
        g = self.config.num_generations
        if not task_ids or len(task_ids) % g or any(len(set(task_ids[i:i+g])) != 1 for i in range(0, len(task_ids), g)):
            raise RuntimeError("Unexpected TRL repeated prompt ordering; single GPU contract violated")
        if getattr(trainer.accelerator, "num_processes", 1) != 1:
            raise RuntimeError("Dynamic sampling requires a single process/GPU")
        if any(p != prompts[i] for i in range(0, len(prompts), g) for p in prompts[i:i+g]):
            raise RuntimeError("Group prompts differ despite matching task IDs")
        model = trainer.accelerator.unwrap_model(trainer.model)
        was_training = model.training
        model.eval()
        results = []
        selected_rewards = []
        batch_id = uuid.uuid4().hex  # Also unique across resume/aborted calls.
        initial_zero = final_zero = exhausted = retries = total_zero = 0
        retry_limit = self.config.dynamic_sampling_max_retries if self.config.dynamic_sampling_enabled else 0
        try:
            for start in range(0, len(task_ids), g):
                task_id = task_ids[start]
                for attempt in range(retry_limit + 1):
                    candidates, rollout_ids = [], []
                    for slot in range(g):
                        rollout_id = f"{batch_id}:{start // g}:{attempt}:{slot}"
                        rollout_ids.append(rollout_id)
                        policy = self.policy_factory() if self.policy_factory else ModelPolicy(model, self.tokenizer)
                        candidates.append(rollout(self.tasks[task_id], policy, self.tokenizer, self.config,
                            verifier=self.verifier, trace_path=self.trace_path,
                            trace_context={"batch_id": batch_id, "rollout_id": rollout_id,
                                           "group_index": start // g, "attempt": attempt,
                                           "selection": "candidate"}))
                    rewards = [r.total for _, r in candidates]
                    if any(not math.isfinite(r) for r in rewards):
                        raise ValueError("Non-finite rollout reward")
                    # TRL stores reward function output in float32. Detect equality
                    # at that precision too, without importing torch on CPU tests.
                    rounded = [struct.unpack("f", struct.pack("f", r))[0] for r in rewards]
                    total_reward_zero = max(rounded) - min(rounded) <= self.config.zero_variance_epsilon
                    outcomes = [primary_outcome(t) for t, _ in candidates]
                    zero = len(set(outcomes)) == 1
                    if attempt == 0:
                        initial_zero += zero
                        total_zero += total_reward_zero
                    accepted = not zero or attempt == retry_limit
                    is_exhausted = accepted and zero and self.config.dynamic_sampling_enabled
                    optimized = optimization_rewards(outcomes, rewards) if accepted else None
                    append_trace(self.trace_path, "rl_group_attempt", {
                        "batch_id": batch_id, "group_index": start // g, "task_id": task_id,
                        "attempt": attempt, "rollout_ids": rollout_ids,
                        "primary_outcomes": [asdict(o) for o in outcomes],
                        "diagnostic_total_reward": rewards, "optimization_reward": optimized,
                        "reward_components": [r.components for _, r in candidates],
                        "correctness_zero_variance": zero,
                        "total_reward_zero_variance": total_reward_zero,
                        "optimization_reward_zeroed": accepted and zero, "exhausted": is_exhausted,
                        "selection": "accepted" if accepted else "discarded"})
                    if accepted:
                        results.extend(candidates)
                        selected_rewards.extend(optimized)
                        final_zero += zero
                        exhausted += is_exhausted
                        break
                    retries += 1
        finally:
            model.train(was_training)
        from cc_agent.rl.evaluation import metrics
        groups = len(task_ids) // g
        self.last_statistics = {
            **metrics(results), "batch_id": batch_id, "group_count": groups,
            "correctness_zero_variance_group_rate": initial_zero / groups,
            "accepted_correctness_zero_variance_group_rate": final_zero / groups,
            "total_reward_zero_variance_group_rate": total_zero / groups,
            "optimization_reward_zeroed_group_rate": final_zero / groups,
            "dynamic_sampling_retry_count": retries,
            "dynamic_sampling_exhausted_rate": exhausted / groups,
            "dynamic_sampling_exhausted_count": exhausted,
            "dynamic_sampling_enabled": self.config.dynamic_sampling_enabled,
        }
        append_trace(self.trace_path, "rl_batch_selected", self.last_statistics)
        append_trace(Path(self.trace_path).parent / "training_log.jsonl", "rl_sampling_metrics", {
            "step": getattr(getattr(trainer, "state", None), "global_step", None),
            **self.last_statistics})
        return {"prompt_ids": [t.prompt_ids for t, _ in results],
                "completion_ids": [t.completion_ids for t, _ in results],
                "env_mask": [t.loss_mask for t, _ in results],
                # TRL snapshots old logprobs before two optimizer iterations reuse this rollout.
                "logprobs": None, "verified_reward": selected_rewards}


def verified_reward(completions, verified_reward, **kwargs):
    if len(completions) != len(verified_reward):
        raise ValueError("Reward/trajectory batch mismatch")
    return verified_reward


def grpo_kwargs(config, output):
    config.validate()
    return dict(output_dir=str(output), loss_type=config.loss_type, epsilon=config.epsilon,
                epsilon_high=config.epsilon_high, mask_truncated_completions=config.mask_truncated_completions, beta=config.beta,
                num_generations=config.num_generations, per_device_train_batch_size=1,
                per_device_eval_batch_size=1, gradient_accumulation_steps=config.num_generations,
                steps_per_generation=config.num_generations, num_iterations=config.num_iterations,
                max_completion_length=config.max_context_tokens, learning_rate=config.learning_rate,
                max_steps=config.max_steps, bf16=True, gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                temperature=1.0, top_p=1.0, top_k=0, scale_rewards="group",
                save_steps=config.save_steps, save_strategy="steps", save_only_model=False,
                logging_steps=1, report_to="none", seed=config.seed, data_seed=config.seed,
                remove_unused_columns=False, use_vllm=False, dataloader_num_workers=0,
                chat_template_kwargs={"enable_thinking": False})


def audit_trainable_adapter(model):
    if set(model.peft_config) != {"default"} or list(model.active_adapters) != ["default"]:
        raise RuntimeError("Exactly one active SFT/RL adapter is required")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names or any("lora_" not in n or "language_model" not in n or
                        any(w in n.lower() for w in ("vision", "visual", "projector")) for n in names):
        raise RuntimeError("Only language-model LoRA parameters may be trainable")
    return names


def trainable_digest(model):
    import hashlib
    import torch
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_model(model_path, adapter, *, trainable):
    from cc_agent.rl.artifacts import adapter_check
    adapter_check(adapter, formal=trainable)
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForMultimodalLM.from_pretrained(model_path, local_files_only=True,
                                                  trust_remote_code=False, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable, local_files_only=True)
    model.to("cuda")
    model.config.use_cache = False
    if trainable:
        model.enable_input_require_grads()
        audit_trainable_adapter(model)
    return model, processor


def train(tasks, config, model_path, adapter, output, resume=None):
    from cc_agent.rl.data import validate_isolation
    validate_isolation(tasks)
    if not tasks or any(t.split != "train" for t in tasks):
        raise ValueError("Trainer accepts training tasks only")
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback
    gradient_steps = []
    class RunLogger(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            # Trainer has restored checkpoint weights before this event.
            audit["initial_sha256"] = trainable_digest(model)
            audit["trainable_parameters"] = audit_trainable_adapter(model)

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            import torch
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
                raise RuntimeError("Missing or non-finite RL LoRA gradients")
            gradient_steps.append({"step": state.global_step, "nonzero": any(torch.count_nonzero(g).item() > 0 for g in gradients)})

        def on_log(self, args, state, control, logs=None, **kwargs):
            append_trace(Path(output) / "training_log.jsonl", "trainer_log", {"step": state.global_step, **(logs or {})})

    check_trl_api()
    model, processor = load_model(model_path, adapter, trainable=True)
    bridge = RolloutBridge(tasks, processor.tokenizer, config, Path(output) / "rollouts.jsonl")
    trainer = GRPOTrainer(model=model, args=GRPOConfig(**grpo_kwargs(config, output)),
                          processing_class=processor, rollout_func=bridge, reward_funcs=verified_reward,
                          callbacks=[RunLogger()],
                          train_dataset=Dataset.from_list([{"prompt": task_prompt(t)} for t in tasks if t.split == "train"]))
    audit = {"trainable_parameters": audit_trainable_adapter(trainer.model),
             "initial_sha256": trainable_digest(trainer.model), "grpo_kwargs": grpo_kwargs(config, output)}
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(str(Path(output) / "final_adapter"))
    trainer.save_state()
    audit.update(final_sha256=trainable_digest(trainer.model), gradient_steps=gradient_steps)
    audit["weights_changed"] = audit["initial_sha256"] != audit["final_sha256"]
    (Path(output) / "model_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
