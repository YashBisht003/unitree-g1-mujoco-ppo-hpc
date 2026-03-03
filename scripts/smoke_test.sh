#!/usr/bin/env bash
set -euo pipefail

# Quick environment sanity check before long HPC runs.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [ ! -d "${VENV_DIR}" ]; then
  echo "ERROR: missing ${VENV_DIR}. Run: bash scripts/bootstrap_env.sh"
  exit 1
fi

source "${VENV_DIR}/bin/activate"
cd "${ROOT_DIR}/mujoco_playground"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ "${USE_CUDA:-1}" != "1" ]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
  export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
  echo "[smoke] forcing CPU backend (USE_CUDA=${USE_CUDA:-0})"
fi

python -c "import jax; print('jax_backend=', jax.default_backend())"
python -c "from mujoco_playground import locomotion; env=locomotion.load('G1JoystickFlatTerrain'); print('env_ok', env.action_size)"

python learning/train_jax_ppo.py \
  --env_name=G1JoystickFlatTerrain \
  --domain_randomization=True \
  --num_timesteps=20000 \
  --num_evals=2 \
  --num_envs=256 \
  --num_eval_envs=64 \
  --seed=1 \
  --suffix=smoke
