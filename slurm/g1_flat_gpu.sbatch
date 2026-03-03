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

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# If your cluster needs module setup, load here.
# Example for Param Ganga docs:
# module load conda-python/3.7

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export USE_CUDA=1

# Reduce if GPU memory is tight.
export NUM_ENVS="${NUM_ENVS:-8192}"
export NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
export SEED="${SEED:-1}"

bash scripts/bootstrap_env.sh
bash scripts/train_g1_flat.sh
