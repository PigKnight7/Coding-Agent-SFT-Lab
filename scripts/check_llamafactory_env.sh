#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-coding_agent_sft}"

conda run --no-capture-output -n "${CONDA_ENV}" python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
print('gpu count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu0', torch.cuda.get_device_name(0))
import triton
print('triton', triton.__version__)
PY

conda run --no-capture-output -n "${CONDA_ENV}" llamafactory-cli env
