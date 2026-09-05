"""Frozen split manifests; training reads only its own split and source repositories."""
import json
from dataclasses import asdict, replace
import hashlib
import random
from pathlib import Path

from cc_agent.rl.environment import stub_solution
from cc_agent.rl.protocol import Task
from cc_agent.rl.hidden import partition_tests


def project_tasks(root):
    root = Path(root).resolve()
    groups = {}
    for split, filename in (("train", "train_alpaca.json"), ("validation", "val_alpaca.json"), ("test", "test_alpaca.json")):
        rows = json.loads((root / "data/llamafactory" / filename).read_text())
        for row in rows:
            group = row["group_id"]
            if group in groups and groups[group] != split:
                raise ValueError("SFT group leakage")
            groups[group] = split
    tasks, excluded = [], []
    for path in sorted((root / "data/tasks").glob("*_tasks.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            group = f"benchmark:{row['source']}:{row['id']}"
            repo = (root / row.get("repo", "__missing__")).resolve()
            try:
                if root not in repo.parents or not repo.is_dir() or group not in groups:
                    raise ValueError("No executable repository or split identity")
                if row.get("test_command") != "pytest -q":
                    raise ValueError("Unsupported verifier command")
                if not list((repo / "tests").glob("test_*.py")):
                    raise ValueError("No executable tests")
                stub_solution((repo / "solution.py").read_text())
                partition_tests(repo)
                tasks.append(Task(row["id"], group, groups[group], str(repo), row["task"]))
            except (ValueError, OSError, SyntaxError) as exc:
                excluded.append({"task_id": row["id"], "reason": str(exc)})
    validate_isolation(tasks)
    return tasks, excluded


def validate_isolation(tasks):
    identities = {}
    for task in tasks:
        if task.split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        for identity in (("task", task.task_id), ("group", task.group_id), ("repo", str(Path(task.repo).resolve()))):
            if identity in identities and identities[identity] != task.split:
                raise ValueError("RL task/group/repository leakage")
            identities[identity] = task.split
    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task ID")


def select(tasks, split):
    validate_isolation(tasks)
    selected = [t for t in tasks if t.split == split]
    if not selected:
        raise ValueError(f"No executable {split} tasks")
    return selected


def grouped_split(tasks, seed=42):
    """Optional plan for a NEW SFT+RL experiment, not safe for an existing SFT adapter."""
    groups = sorted({t.group_id for t in tasks})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    assignments = {g: ("train" if i < int(n * .8) else "validation" if i < int(n * .9) else "test")
                   for i, g in enumerate(groups)}
    result = [replace(t, split=assignments[t.group_id]) for t in tasks]
    validate_isolation(result)
    return result


def freeze_tasks(root):
    """Explicit offline preparation only; never called by training."""
    root = Path(root).resolve()
    tasks, excluded = project_tasks(root)
    destination = root / "data/rl"
    destination.mkdir(parents=True, exist_ok=True)
    index = {"schema_version": 1, "seed": 42, "split_policy": "preserve_sft_holdouts",
             "excluded": excluded, "splits": {}, "identities": [], "hidden_assertions": {},
             "sft_train_sha256": hashlib.sha256((root / "data/llamafactory/train_alpaca.json").read_bytes()).hexdigest()}
    for split in ("train", "validation", "test"):
        rows = []
        public = hidden = 0
        for t in sorted(select(tasks, split), key=lambda t: t.group_id):
            part = partition_tests(t.repo)
            public += part.public_count
            hidden += part.hidden_count
            row = {**asdict(t), "repo": str(Path(t.repo).relative_to(root))}
            row["source_hashes"] = {p.relative_to(t.repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [Path(t.repo) / "solution.py", *sorted((Path(t.repo) / "tests").glob("test_*.py"))]}
            rows.append(row)
            index["identities"].append({k: row[k] for k in ("task_id", "group_id", "repo", "split")})
        path = destination / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        index["splits"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
        index["hidden_assertions"][split] = {"tasks": len(rows), "public": public, "hidden": hidden}
    proposal = grouped_split(tasks)
    source_train = {t.group_id for t in tasks if t.split == "train"}
    index["rebalance_audit"] = {"seed": 42, "counts": {s: sum(t.split == s for t in proposal) for s in index["splits"]},
                                "sft_seen_holdout_groups": sorted(t.group_id for t in proposal if t.split != "train" and t.group_id in source_train),
                                "requires_new_sft_split_and_adapter": True}
    (destination / "manifest.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    return index


def load_tasks(root, split=None):
    root = Path(root).resolve()
    index = json.loads((root / "data/rl/manifest.json").read_text())
    identities = [Task(**row, instruction="") for row in index["identities"]]
    validate_isolation(identities)
    tasks = []
    expected_identities = {(r["task_id"], r["group_id"], r["repo"], r["split"]) for r in index["identities"]}
    for name in ([split] if split else ["train", "validation", "test"]):
        if name not in index["splits"]:
            raise ValueError("Unknown split")
        path = root / "data/rl" / f"{name}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != index["splits"][name]:
            raise ValueError("RL split manifest changed; rerun explicit data audit")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if tuple(row[k] for k in ("task_id", "group_id", "repo", "split")) not in expected_identities:
                raise ValueError("Task not in audited group index")
            original_hashes = row.pop("source_hashes")
            row["repo"] = str((root / row["repo"]).resolve())
            if root not in Path(row["repo"]).parents or row["split"] != name:
                raise ValueError("Invalid frozen task identity")
            for rel, expected in original_hashes.items():
                source = (Path(row["repo"]) / rel).resolve()
                if Path(row["repo"]) not in source.parents or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise ValueError("Original task/test changed since split audit")
            row["editable"] = tuple(row["editable"])
            tasks.append(Task(**row))
    validate_isolation(tasks)
    return tasks, index["excluded"]
