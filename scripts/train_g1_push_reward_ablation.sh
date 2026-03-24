#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${ROUGH_CKPT:-}" ]; then
  echo "ERROR: ROUGH_CKPT is required."
  echo "Set it to the common rough-stage checkpoint for the reward ablation."
  exit 1
fi

ABLATION_MODE="${ABLATION_MODE:-${PUSH_REWARD_ABLATION_MODE:-survival_min}}"
PROFILE="${PROFILE:-single_gpu}"
SEED="${SEED:-1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

case "${ABLATION_MODE}" in
  survival_min|survival_stable|survival_phi) ;;
  *)
    echo "ERROR: invalid ABLATION_MODE=${ABLATION_MODE}. Expected survival_min|survival_stable|survival_phi"
    exit 1
    ;;
esac

SUFFIX_DEFAULT="g1-push-${PROFILE}-${ABLATION_MODE}-seed${SEED}-${TIMESTAMP}"

env \
  PUSH_REWARD_MODE="${PUSH_REWARD_MODE:-off}" \
  PUSH_REWARD_ABLATION_MODE="${ABLATION_MODE}" \
  SUFFIX="${SUFFIX:-${SUFFIX_DEFAULT}}" \
  NUM_TIMESTEPS="${NUM_TIMESTEPS:-25000000}" \
  NUM_ENVS="${NUM_ENVS:-1024}" \
  NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}" \
  SEED="${SEED}" \
  USE_TB="${USE_TB:-1}" \
  USE_WANDB="${USE_WANDB:-0}" \
  USE_CUDA="${USE_CUDA:-1}" \
  MODE="${MODE:-ppo_dense}" \
  ROUGH_CKPT="${ROUGH_CKPT}" \
  PLAYGROUND_CONFIG_OVERRIDES="${PLAYGROUND_CONFIG_OVERRIDES:-}" \
  DRY_RUN="${DRY_RUN:-0}" \
  bash "${ROOT_DIR}/scripts/train_g1_push_recovery.sh"
