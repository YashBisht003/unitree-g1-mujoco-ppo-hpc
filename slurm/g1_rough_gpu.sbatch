#!/bin/bash
#SBATCH --job-name=g1_rough_ppo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# If your cluster needs module setup, load here.
# module load conda-python/3.7

if [ -z "${FLAT_CKPT:-}" ]; then
  echo "ERROR: FLAT_CKPT is required for rough-terrain finetuning."
  echo "Submit with: sbatch --export=ALL,FLAT_CKPT=/abs/path/to/checkpoints slurm/g1_rough_gpu.sbatch"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export USE_CUDA=1

export NUM_ENVS="${NUM_ENVS:-8192}"
export NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
export SEED="${SEED:-1}"

bash scripts/bootstrap_env.sh
bash scripts/train_g1_rough_from_flat.sh
