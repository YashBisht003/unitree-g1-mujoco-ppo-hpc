#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="${ROOT_DIR}/slurm/g1_flat_gpu.sbatch"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on the HPC login node."
  exit 1
fi

PARTITION="${PARTITION:-gpu}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

EXPORT_VARS=(
  "ALL"
  "NUM_ENVS=${NUM_ENVS:-8192}"
  "NUM_EVAL_ENVS=${NUM_EVAL_ENVS:-128}"
  "NUM_TIMESTEPS=${NUM_TIMESTEPS:-200000000}"
  "SEED=${SEED:-1}"
  "USE_CUDA=${USE_CUDA:-1}"
  "PLAYGROUND_REF=${PLAYGROUND_REF:-f2159f3}"
  "PYTHON_BIN=${PYTHON_BIN:-python3}"
  "USE_TB=${USE_TB:-1}"
  "USE_WANDB=${USE_WANDB:-0}"
)

MEM_PRINT="${MEM:-<cluster-default>}"
echo "[submit-flat] partition=${PARTITION} gpus=${GPUS} cpus=${CPUS_PER_TASK} mem=${MEM_PRINT} time=${TIME_LIMIT}"
echo "[submit-flat] script=${SBATCH_SCRIPT}"

# Ensure stdout/stderr target directory exists before Slurm opens log files.
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
