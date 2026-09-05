from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import struct


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def adapter_check(path, *, formal=False):
    path = Path(path).resolve()
    config = json.loads((path / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA" or not (path / "adapter_model.safetensors").is_file():
        raise ValueError("Expected a saved LoRA safetensors adapter")
    targets = config.get("target_modules", [])
    if not targets or config.get("modules_to_save") or any(word in str(targets).lower() for word in ("visual", "vision", "projector")):
        raise ValueError("Expected language-only LoRA without modules_to_save")
    with (path / "adapter_model.safetensors").open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length <= 16 * 1024 * 1024:
            raise ValueError("Invalid safetensors header")
        header = json.loads(stream.read(length))
    names = [name for name in header if name != "__metadata__"]
    if not names or any("lora_" not in n or "language_model" not in n or any(w in n.lower() for w in ("vision", "visual", "projector")) for n in names):
        raise ValueError("Adapter tensors must be language-model LoRA only")
    if formal:
        manifest = json.loads((path / "experiment_manifest.json").read_text())
        if manifest.get("run_kind") != "formal":
            raise ValueError("RL must start from formal SFT, not Smoke")
        training = manifest.get("training", {})
        if training.get("freeze_vision_tower") is not True or training.get("freeze_multi_modal_projector") is not True:
            raise ValueError("SFT manifest must prove frozen vision/projector")
        split_index = json.loads((Path(__file__).resolve().parents[3] / "data/rl/manifest.json").read_text())
        if manifest.get("artifacts", {}).get("datasets", {}).get("data/llamafactory/train_alpaca.json") != split_index["sft_train_sha256"]:
            raise ValueError("SFT training split provenance does not match audited RL holdouts")
        state = json.loads((path / "trainer_state.json").read_text())
        if state.get("global_step", 0) < 1 or state.get("epoch", 0) < 1:
            raise ValueError("Formal SFT completion is not proven")
    return path


def manifest(root, tasks, config, model, adapter, kind):
    root = Path(root)
    paths = []
    for directory in ("src", "scripts", "configs"):
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths += list(root.glob("requirements*.txt"))
    paths += [root / "data/rl/manifest.json"]
    paths += [root / "data/rl" / f"{split}.jsonl" for split in sorted({t.split for t in tasks})]
    dependencies = {}
    for name in ("trl", "transformers", "peft", "torch", "datasets", "accelerate"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not-installed"
    return {"schema_version": 1, "kind": kind, "config": asdict(config),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "dependencies": dependencies,
            "files": {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)},
            "tasks": [{**asdict(t), "editable": list(t.editable), "repo": str(Path(t.repo).relative_to(root)),
                       "hidden_hashes": {p.relative_to(t.hidden_tests).as_posix(): digest(p)
                                         for p in sorted(Path(t.hidden_tests).rglob("*")) if p.is_file()} if t.hidden_tests else {},
                       "repo_hashes": {p.relative_to(t.repo).as_posix(): digest(p) for p in sorted(Path(t.repo).rglob("*"))
                                       if p.is_file() and "__pycache__" not in p.parts}} for t in tasks],
            "model_config_sha256": digest(Path(model) / "config.json"),
            "model_files": {p.name: digest(p) for p in sorted(Path(model).iterdir())
                            if p.is_file() and (p.suffix in {".safetensors", ".json", ".jinja", ".txt", ".model"})},
            "sft_adapter_sha256": digest(Path(adapter) / "adapter_model.safetensors"),
            "sft_adapter_config_sha256": digest(Path(adapter) / "adapter_config.json")}


def prepare_run(output, record, resume=None):
    output = Path(output).resolve()
    destination = output / "rl_manifest.json"
    if resume:
        checkpoint = Path(resume).resolve()
        if checkpoint.parent != output or not checkpoint.name.startswith("checkpoint-"):
            raise ValueError("Resume checkpoint must belong to this run")
        if json.loads(destination.read_text()) != record:
            raise ValueError("Resume manifest mismatch: code/data/config/adapter/runtime changed")
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
            if not (checkpoint / name).is_file():
                raise ValueError(f"Incomplete checkpoint: {name}")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory must be empty; use resume for an existing run")
        output.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
