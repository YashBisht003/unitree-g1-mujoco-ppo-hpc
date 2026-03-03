#!/bin/bash
#SBATCH --job-name=g1_flat_ppo_cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 022

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
chmod 755 logs || true

# Sanitize inherited submit-shell conda/python vars.
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE __PYVENV_LAUNCHER__ || true
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL _CE_M _CE_CONDA || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export USE_CUDA=0
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
echo "[slurm-cpu] forcing CPU backend."

export NUM_ENVS="${NUM_ENVS:-512}"
export NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-64}"
export SEED="${SEED:-1}"

bash scripts/bootstrap_env.sh
bash scripts/train_g1_flat.sh
