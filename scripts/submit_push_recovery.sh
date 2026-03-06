#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on the HPC login node."
  exit 1
fi

if [ -z "${ROUGH_CKPT:-}" ]; then
  echo "ERROR: ROUGH_CKPT is required."
  echo "Try:"
  echo "  export ROUGH_CKPT=\"$(cd "${ROOT_DIR}" && pwd)/mujoco_playground/logs/<rough_run>/checkpoints/<step_dir>\""
  exit 1
fi

MODE="${MODE:-ppo_dense}" # ppo_dense|maxrl_binary|maxrl_t
USE_CUDA="${USE_CUDA:-1}"
if [ "${USE_CUDA}" != "1" ]; then
  echo "ERROR: submit_push_recovery currently supports GPU path only (USE_CUDA=1)."
  exit 1
fi

case "${MODE}" in
  ppo_dense)
    SBATCH_SCRIPT="${ROOT_DIR}/slurm/g1_push_recovery_gpu.sbatch"
    ;;
  maxrl_binary)
    SBATCH_SCRIPT="${ROOT_DIR}/slurm/g1_push_maxrl_gpu.sbatch"
    ;;
  maxrl_t)
    SBATCH_SCRIPT="${ROOT_DIR}/slurm/g1_push_maxrl_t_gpu.sbatch"
    ;;
  *)
    echo "ERROR: invalid MODE=${MODE}. Expected ppo_dense|maxrl_binary|maxrl_t"
    exit 1
    ;;
esac

PARTITION="${PARTITION:-gpu}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

EXPORT_VARS=(
  "ALL"
  "MODE=${MODE}"
  "ADV_MODE=${ADV_MODE:-}"
  "ROUGH_CKPT=${ROUGH_CKPT}"
  "ENV_NAME=${ENV_NAME:-G1JoystickRoughTerrain}"
  "NUM_ENVS=${NUM_ENVS:-1024}"
  "NUM_EVAL_ENVS=${NUM_EVAL_ENVS:-128}"
  "NUM_TIMESTEPS=${NUM_TIMESTEPS:-50000000}"
  "SEED=${SEED:-1}"
  "SUFFIX=${SUFFIX:-g1-push-${MODE}-seed${SEED:-1}}"
  "PLAYGROUND_CONFIG_OVERRIDES=${PLAYGROUND_CONFIG_OVERRIDES:-}"
  "SCENARIO_GROUP_SIZE=${SCENARIO_GROUP_SIZE:-8}"
  "MAXRL_LOG_ONLY=${MAXRL_LOG_ONLY:-0}"
  "MAXRL_SCENARIO_KEY=${MAXRL_SCENARIO_KEY:-push_cfg}"
  "MAXRL_VERBOSE=${MAXRL_VERBOSE:-1}"
  "PUSH_ADV_MASK_MODE=${PUSH_ADV_MASK_MODE:-off}"
  "PUSH_ADV_PRE_WEIGHT=${PUSH_ADV_PRE_WEIGHT:-0.1}"
  "PUSH_EVENT_EPS=${PUSH_EVENT_EPS:-1e-6}"
  "PUSH_ENTROPY_MODE=${PUSH_ENTROPY_MODE:-off}"
  "PUSH_ENTROPY_DELTA=${PUSH_ENTROPY_DELTA:-0.0}"
  "PUSH_REWARD_MODE=${PUSH_REWARD_MODE:-off}"
  "RECOVERY_WINDOW_K=${RECOVERY_WINDOW_K:-60}"
  "RECOVERY_WINDOW_TRACKING_SCALE=${RECOVERY_WINDOW_TRACKING_SCALE:-0.2}"
  "RECOVERY_OMEGA_WEIGHT=${RECOVERY_OMEGA_WEIGHT:-0.05}"
  "RECOVERY_BONUS=${RECOVERY_BONUS:-8.0}"
  "RECOVERY_BONUS_STABILITY_STEPS=${RECOVERY_BONUS_STABILITY_STEPS:-10}"
  "RECOVERY_BONUS_DELAY_STEPS=${RECOVERY_BONUS_DELAY_STEPS:-10}"
  "RECOVERY_STABLE_LIN_MIN=${RECOVERY_STABLE_LIN_MIN:-0.7}"
  "RECOVERY_STABLE_ANG_MIN=${RECOVERY_STABLE_ANG_MIN:-0.7}"
  "CAPTURE_POINT_LOG=${CAPTURE_POINT_LOG:-0}"
  "USE_CUDA=${USE_CUDA}"
  "JAX_CUDA_EXTRA=${JAX_CUDA_EXTRA:-cuda12}"
  "BOOTSTRAP_OFFLINE=${BOOTSTRAP_OFFLINE:-1}"
  "PLAYGROUND_REF=${PLAYGROUND_REF:-d886c80}"
  "PYTHON_BIN=${PYTHON_BIN:-python3}"
  "USE_TB=${USE_TB:-1}"
  "USE_WANDB=${USE_WANDB:-0}"
)

MEM_PRINT="${MEM:-<cluster-default>}"
echo "[submit-push] mode=${MODE} partition=${PARTITION} gpus=${GPUS} cpus=${CPUS_PER_TASK} mem=${MEM_PRINT} time=${TIME_LIMIT}"
echo "[submit-push] rough_ckpt=${ROUGH_CKPT}"
echo "[submit-push] script=${SBATCH_SCRIPT}"

mkdir -p "${ROOT_DIR}/logs"

SBATCH_ARGS=(
  "--partition=${PARTITION}"
  "--gres=gpu:${GPUS}"
  "--cpus-per-task=${CPUS_PER_TASK}"
  "--time=${TIME_LIMIT}"
  "--chdir=${ROOT_DIR}"
  "--export=$(IFS=,; echo "${EXPORT_VARS[*]}")"
)

if [ -n "${MEM}" ]; then
  SBATCH_ARGS+=("--mem=${MEM}")
fi

sbatch "${SBATCH_ARGS[@]}" "${SBATCH_SCRIPT}"
