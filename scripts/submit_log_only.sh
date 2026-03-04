#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ -z "${ROUGH_CKPT:-}" ]; then
  echo "ERROR: ROUGH_CKPT is required."
  echo "Try: export ROUGH_CKPT=\"$(bash scripts/latest_rough_checkpoint.sh)\""
  exit 1
fi

export MODE="${MODE:-ppo_dense}"
export ADV_MODE="${ADV_MODE:-maxrl_binary}"
export SCENARIO_GROUP_SIZE="${SCENARIO_GROUP_SIZE:-8}"
export MAXRL_LOG_ONLY="${MAXRL_LOG_ONLY:-1}"
export NUM_ENVS="${NUM_ENVS:-1024}"
export NUM_TIMESTEPS="${NUM_TIMESTEPS:-5000000}"
export PARTITION="${PARTITION:-gpu}"
export MAXRL_VERBOSE="${MAXRL_VERBOSE:-1}"

echo "[submit-log-only] ROUGH_CKPT=${ROUGH_CKPT}"
echo "[submit-log-only] MODE=${MODE} ADV_MODE=${ADV_MODE} GROUP=${SCENARIO_GROUP_SIZE} LOG_ONLY=${MAXRL_LOG_ONLY}"

bash scripts/submit_push_recovery.sh
