#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/qwen3_5_2b_smoke.yaml}"
FORMAL_CONFIG="${FORMAL_CONFIG:-configs/qwen3_5_2b_lora_sft.yaml}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-outputs/qwen3_5_2b_smoke_lora}"
FORMAL_OUTPUT_DIR="${FORMAL_OUTPUT_DIR:-outputs/qwen3_5_2b_lora_sft}"
CPU_ONLY="${CPU_ONLY:-0}"

run_config_check() {
  local config_path="$1"
  local output_path="$2"
  if [[ "${CPU_ONLY}" == "1" ]]; then
    "${PYTHON_BIN}" scripts/check_sft_environment.py --config "${config_path}" --static-only
  else
    : "${MODEL_PATH:?Set MODEL_PATH to the complete local Qwen3.5-2B directory}"
    "${PYTHON_BIN}" scripts/check_sft_environment.py \
      --config "${config_path}" \
      --model-path "${MODEL_PATH}" \
      --output-dir "${output_path}"
  fi
}

echo "[preflight] mode=$([[ "${CPU_ONLY}" == "1" ]] && echo cpu-only || echo full)"
echo "[preflight] Smoke configuration and runtime"
run_config_check "${SMOKE_CONFIG}" "${SMOKE_OUTPUT_DIR}"

echo "[preflight] Formal configuration and runtime"
run_config_check "${FORMAL_CONFIG}" "${FORMAL_OUTPUT_DIR}"

echo "[preflight] Strict SFT data validation"
"${PYTHON_BIN}" scripts/validate_sft_data.py --strict --require-explicit-task-type

echo "[preflight] Unit tests"
PYTHONPATH=src "${PYTHON_BIN}" -m unittest discover -s tests -v

echo "[preflight] Python compilation"
mapfile -d '' python_files < <(find src scripts tests -type f -name '*.py' -print0)
"${PYTHON_BIN}" -m py_compile "${python_files[@]}"

echo "[preflight] Shell syntax"
while IFS= read -r -d '' shell_file; do
  bash -n "${shell_file}"
done < <(find scripts -type f -name '*.sh' -print0)

echo "[preflight] Git whitespace"
git diff --check

if [[ "${CPU_ONLY}" == "1" ]]; then
  echo "[preflight] CPU checks passed; GPU/model checks remain pending."
else
  echo "[preflight] READY: all static, data, test, CUDA, model, template and LoRA checks passed."
fi
