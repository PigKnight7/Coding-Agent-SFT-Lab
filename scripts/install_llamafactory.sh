#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-coding_agent_sft}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
PIP_TIMEOUT="${PIP_TIMEOUT:-180}"
PIP_RETRIES="${PIP_RETRIES:-10}"

PIP_ARGS=(
  --index-url "${PIP_INDEX_URL}"
  --trusted-host "${PIP_TRUSTED_HOST}"
  --timeout "${PIP_TIMEOUT}"
  --retries "${PIP_RETRIES}"
  --prefer-binary
  --no-cache-dir
)

echo "Installing LLaMA-Factory in conda env: ${CONDA_ENV}"
echo "Using PyPI mirror: ${PIP_INDEX_URL}"

conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install -U pip setuptools wheel "${PIP_ARGS[@]}"
conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install -U "llamafactory[torch,metrics]" "${PIP_ARGS[@]}"

# Some mirrors may leave a broken Triton wheel, which causes segmentation faults
# when LLaMA-Factory imports torch/triton. Force reinstall the torch-matched version.
conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install --force-reinstall triton==3.2.0 "${PIP_ARGS[@]}"

conda run --no-capture-output -n "${CONDA_ENV}" python -c "import triton; print('triton', triton.__version__)"
conda run --no-capture-output -n "${CONDA_ENV}" llamafactory-cli env
