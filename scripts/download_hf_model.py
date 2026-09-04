"""Legacy generic downloader; it is not part of the recommended Qwen3.5-2B workflow."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustly pre-download a HuggingFace model with mirror and low concurrency.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B", help="HF repo id, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--local-dir", default="models/Qwen3-8B", help="Local output directory")
    parser.add_argument("--endpoint", default="https://hf-mirror.com", help="HF endpoint mirror")
    parser.add_argument("--max-workers", type=int, default=1, help="Lower is more stable on weak networks")
    parser.add_argument("--include", nargs="*", default=None, help="Optional allow patterns")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", args.endpoint)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    from huggingface_hub import snapshot_download

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"[download] model_id      : {args.model_id}", flush=True)
    print(f"[download] local_dir     : {local_dir.resolve()}", flush=True)
    print(f"[download] HF_ENDPOINT   : {os.environ.get('HF_ENDPOINT')}", flush=True)
    print(f"[download] max_workers   : {args.max_workers}", flush=True)
    print(f"[download] include       : {args.include or 'all files'}", flush=True)
    print(f"[download] started_at    : {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)

    before = _snapshot_local_files(local_dir)
    if before:
        print(f"[download] existing files: {len(before)}", flush=True)
        _print_local_summary(local_dir)

    kwargs = {
        "repo_id": args.model_id,
        "local_dir": str(local_dir),
        "max_workers": args.max_workers,
        "local_dir_use_symlinks": False,
    }
    if args.include:
        kwargs["allow_patterns"] = args.include

    path = snapshot_download(**kwargs)

    after = _snapshot_local_files(local_dir)
    new_files = sorted(set(after) - set(before))
    print("=" * 80, flush=True)
    print(f"[download] downloaded_to : {path}", flush=True)
    print(f"[download] finished_at   : {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[download] total_files   : {len(after)}", flush=True)
    print(f"[download] new_files     : {len(new_files)}", flush=True)
    if new_files:
        print("[download] new file list:", flush=True)
        for file in new_files[:50]:
            print(f"  + {file}", flush=True)
        if len(new_files) > 50:
            print(f"  ... {len(new_files) - 50} more", flush=True)
    _print_local_summary(local_dir)
    print("=" * 80, flush=True)


def _snapshot_local_files(local_dir: Path) -> list[str]:
    if not local_dir.exists():
        return []
    return [str(path.relative_to(local_dir)) for path in local_dir.rglob("*") if path.is_file()]


def _print_local_summary(local_dir: Path) -> None:
    files = [path for path in local_dir.rglob("*") if path.is_file()]
    total_size = sum(path.stat().st_size for path in files)
    safetensors = [path for path in files if path.name.endswith(".safetensors")]
    print(f"[download] local size    : {_human_size(total_size)}", flush=True)
    print(f"[download] safetensors   : {len(safetensors)}", flush=True)
    for path in sorted(safetensors)[:20]:
        print(f"  - {path.name}: {_human_size(path.stat().st_size)}", flush=True)


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f}{unit}"
        value /= 1024
    return f"{size}B"


if __name__ == "__main__":
    main()
