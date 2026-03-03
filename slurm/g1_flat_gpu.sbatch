#!/bin/bash
#SBATCH --job-name=g1_flat_ppo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 022

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
chmod 755 logs || true

# If your cluster needs module setup, load here.
# Example for Param Ganga docs:
# module load conda-python/3.7

# Sanitize inherited submit-shell conda/python vars.
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE __PYVENV_LAUNCHER__ || true
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL _CE_M _CE_CONDA || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export USE_CUDA="${USE_CUDA:-1}"

# CUDA12 JAX requires a sufficiently new NVIDIA driver (major >= 525).
# Auto-fallback to CPU on older driver stacks to avoid hard crashes.
if [ "${USE_CUDA}" = "1" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
    DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
    if echo "${DRIVER_MAJOR}" | grep -Eq '^[0-9]+$' && [ "${DRIVER_MAJOR}" -lt 525 ]; then
      echo "[slurm] CUDA driver ${DRIVER_VERSION} too old for jax[cuda12]; forcing CPU backend."
      export USE_CUDA=0
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

# Reduce if GPU memory is tight.
export NUM_ENVS="${NUM_ENVS:-8192}"
export NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
export SEED="${SEED:-1}"

bash scripts/bootstrap_env.sh
bash scripts/train_g1_flat.sh
