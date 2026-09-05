#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/resume_agentic_rl.sh CHECKPOINT --model-path MODEL --sft-adapter ADAPTER --output-dir RUN [--config CONFIG]"
  exit 0
fi
: "${1:?Supply the exact checkpoint directory as first argument}"
RL_CHECKPOINT="$1"
shift
exec "${PYTHON_BIN:-python3}" scripts/agentic_rl.py train --resume "${RL_CHECKPOINT}" "$@"
