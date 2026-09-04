#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
CONFIG_PATH="${CONFIG_PATH:-configs/qwen3_5_2b_smoke.yaml}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the complete local Qwen3.5-2B directory}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3_5_2b_smoke_lora}"
FORMAL_OUTPUT_DIR="${FORMAL_OUTPUT_DIR:-outputs/qwen3_5_2b_lora_sft}"

if [[ "${OUTPUT_DIR}" == "${FORMAL_OUTPUT_DIR}" ]]; then
  echo "Smoke and formal output directories must be different." >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/check_sft_environment.py \
  --config "${CONFIG_PATH}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/create_experiment_manifest.py \
  --config "${CONFIG_PATH}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-kind smoke

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "${LLAMAFACTORY_CLI}" train "${CONFIG_PATH}" \
  model_name_or_path="${MODEL_PATH}" \
  output_dir="${OUTPUT_DIR}" \
  overwrite_output_dir=true
