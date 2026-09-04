#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Legacy Qwen3-8B downloader. The Qwen3.5-2B runbook expects MODEL_PATH to be prepared externally.
CONDA_ENV="${CONDA_ENV:-coding_agent_sft}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
LOCAL_DIR="${LOCAL_DIR:-models/Qwen3-8B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MAX_WORKERS="${MAX_WORKERS:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"

export HF_ENDPOINT
export HF_HOME="${HF_HOME:-.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/download_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "${LOCAL_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}"

free_kb=$(df -Pk "${LOCAL_DIR}" | awk 'NR==2 {print $4}')
free_gb=$((free_kb / 1024 / 1024))
if (( free_gb < MIN_FREE_GB )); then
  echo "ERROR: Not enough free disk space for LOCAL_DIR=${LOCAL_DIR}" | tee -a "${LOG_FILE}"
  echo "Available: ${free_gb}GB, required at least: ${MIN_FREE_GB}GB" | tee -a "${LOG_FILE}"
  echo "Set LOCAL_DIR to a larger filesystem, for example:" | tee -a "${LOG_FILE}"
  echo "  LOCAL_DIR=/path/to/models/Qwen3-8B bash scripts/download_qwen_model.sh" | tee -a "${LOG_FILE}"
  exit 1
fi

echo "Pre-downloading model before training" | tee -a "${LOG_FILE}"
echo "MODEL_ID=${MODEL_ID}" | tee -a "${LOG_FILE}"
echo "LOCAL_DIR=${LOCAL_DIR}" | tee -a "${LOG_FILE}"
echo "HF_HOME=${HF_HOME}" | tee -a "${LOG_FILE}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE}" | tee -a "${LOG_FILE}"
echo "HF_ENDPOINT=${HF_ENDPOINT}" | tee -a "${LOG_FILE}"
echo "MAX_WORKERS=${MAX_WORKERS}" | tee -a "${LOG_FILE}"
echo "FREE_SPACE=${free_gb}GB" | tee -a "${LOG_FILE}"
echo "LOG_FILE=${LOG_FILE}" | tee -a "${LOG_FILE}"

set -o pipefail
conda run --no-capture-output -n "${CONDA_ENV}" python scripts/download_hf_model.py \
  --model-id "${MODEL_ID}" \
  --local-dir "${LOCAL_DIR}" \
  --endpoint "${HF_ENDPOINT}" \
  --max-workers "${MAX_WORKERS}" 2>&1 | tee -a "${LOG_FILE}"

echo "Model local path: ${LOCAL_DIR}" | tee -a "${LOG_FILE}"
echo "Download log: ${LOG_FILE}"
