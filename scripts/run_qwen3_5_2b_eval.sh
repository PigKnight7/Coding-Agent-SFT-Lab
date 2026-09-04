#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen/Qwen3.5-2B directory}"
: "${ADAPTER_DIR:?Set ADAPTER_DIR to the completed formal LoRA output directory}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3_5_2b_base_vs_sft_eval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" scripts/eval_before_after_sft.py \
  --model-path "${MODEL_PATH}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed 42 \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}"
