#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" scripts/check_rl_sandbox.py
CPU_ONLY=1 bash scripts/final_preflight.sh
bash scripts/check_rl_environment.sh --static-only --config configs/agentic_rl_smoke.yaml
bash scripts/check_rl_environment.sh --static-only --config configs/agentic_rl_train.yaml
while IFS= read -r -d '' rl_file; do
  rl_status=0
  git diff --no-index --check /dev/null "${rl_file}" || rl_status=$?
  # --no-index returns 1 for an added file even when whitespace checks pass.
  if (( rl_status > 1 )); then
    exit "${rl_status}"
  fi
done < <(git ls-files --others --exclude-standard -z)
echo "[PASS] Agentic RL CPU-only final preflight; no GPU/GRPO/model evaluation was run."
