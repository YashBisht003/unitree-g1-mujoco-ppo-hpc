#!/bin/bash
#SBATCH --job-name=g1_push_maxrl_t
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 000

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
chmod 777 logs || true

unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE __PYVENV_LAUNCHER__ || true
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL _CE_M _CE_CONDA || true

if [ -z "${ROUGH_CKPT:-}" ]; then
  echo "ERROR: ROUGH_CKPT is required for push-recovery finetuning."
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export USE_CUDA="${USE_CUDA:-1}"
export JAX_CUDA_EXTRA="${JAX_CUDA_EXTRA:-cuda12}"

if [ "${USE_CUDA}" = "1" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
    DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
    MIN_DRIVER_MAJOR=525
    if [ "${JAX_CUDA_EXTRA}" = "cuda11_pip" ]; then
      MIN_DRIVER_MAJOR=0
    fi
    if [ "${MIN_DRIVER_MAJOR}" -gt 0 ] && echo "${DRIVER_MAJOR}" | grep -Eq '^[0-9]+$' && [ "${DRIVER_MAJOR}" -lt "${MIN_DRIVER_MAJOR}" ]; then
      echo "[slurm] CUDA driver ${DRIVER_VERSION} too old for jax[${JAX_CUDA_EXTRA}] (need >=${MIN_DRIVER_MAJOR}); forcing CPU backend."
      export USE_CUDA=0
    else
      echo "[slurm] CUDA driver ${DRIVER_VERSION} accepted for jax[${JAX_CUDA_EXTRA}]."
    fi
  else
    echo "[slurm] nvidia-smi not found on node; forcing CPU backend."
    export USE_CUDA=0
  fi
fi

if [ "${USE_CUDA}" != "1" ]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
  export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
  echo "[slurm] USE_CUDA=${USE_CUDA}; using CPU backend."
fi

export MODE="${MODE:-maxrl_t}"
export NUM_ENVS="${NUM_ENVS:-1024}"
export NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
export SEED="${SEED:-1}"

bash scripts/bootstrap_env.sh
bash scripts/train_g1_push_recovery.sh
