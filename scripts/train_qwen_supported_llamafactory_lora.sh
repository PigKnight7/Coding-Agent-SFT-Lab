#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Legacy Qwen3-8B entrypoint retained for reproducibility. Use TRAINING_RUNBOOK.md for Qwen3.5-2B.
CONDA_ENV="${CONDA_ENV:-coding_agent_sft}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Use a Transformers/LLaMA-Factory supported Qwen model for the first runnable SFT smoke test.
# Recommended alternatives:
#   Qwen/Qwen2.5-Coder-7B-Instruct
#   Qwen/Qwen2.5-7B-Instruct
#   Qwen/Qwen3-8B
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-}"
if [[ -n "${LOCAL_MODEL_DIR}" && -d "${LOCAL_MODEL_DIR}" ]]; then
  MODEL_ID="${LOCAL_MODEL_DIR}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen_supported_coding_agent_lora_llamafactory}"
DATASET_DIR="${DATASET_DIR:-data/llamafactory}"

conda run --no-capture-output -n "${CONDA_ENV}" python scripts/prepare_llamafactory_sft.py

conda run --no-capture-output -n "${CONDA_ENV}" llamafactory-cli train \
  --stage sft \
  --do_train true \
  --model_name_or_path "${MODEL_ID}" \
  --trust_remote_code true \
  --dataset_dir "${DATASET_DIR}" \
  --dataset coding_agent_train \
  --eval_dataset coding_agent_val \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_target all \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir true \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --save_steps 20 \
  --eval_steps 20 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --max_samples 100000 \
  --cutoff_len 4096 \
  --bf16 true \
  --plot_loss true
