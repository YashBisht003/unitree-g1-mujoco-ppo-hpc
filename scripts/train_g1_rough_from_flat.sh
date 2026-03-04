#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PLAYGROUND_DIR="${ROOT_DIR}/mujoco_playground"

if [ ! -d "${VENV_DIR}" ]; then
  echo "ERROR: missing ${VENV_DIR}. Run: bash scripts/bootstrap_env.sh"
  exit 1
fi
if [ ! -d "${PLAYGROUND_DIR}" ]; then
  echo "ERROR: missing ${PLAYGROUND_DIR}. Run: bash scripts/bootstrap_env.sh"
  exit 1
fi

if [ -z "${FLAT_CKPT:-}" ]; then
  echo "ERROR: FLAT_CKPT is not set."
  echo "Set it to the flat run checkpoint directory, for example:"
  echo "  export FLAT_CKPT=/abs/path/to/mujoco_playground/logs/<flat_run>/checkpoints"
  exit 1
fi

# Prevent inherited conda/python env vars from polluting this venv.
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE __PYVENV_LAUNCHER__ || true
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL _CE_M _CE_CONDA || true

source "${VENV_DIR}/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${VENV_DIR}/.pycache"
umask 000
if [ "${USE_CUDA:-1}" = "1" ]; then
  VENV_SITE="$("${VENV_DIR}/bin/python" -B - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
  if [ -d "${VENV_SITE}/nvidia" ]; then
    CUDA_LIB_DIRS="$(find "${VENV_SITE}/nvidia" -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')"
    if [ -n "${CUDA_LIB_DIRS}" ]; then
      export LD_LIBRARY_PATH="${CUDA_LIB_DIRS%:}:${LD_LIBRARY_PATH:-}"
      echo "[train-rough] added CUDA libs from ${VENV_SITE}/nvidia to LD_LIBRARY_PATH"
    fi
  fi
fi
cd "${PLAYGROUND_DIR}"
mkdir -p logs
chmod 777 logs || true

NUM_TIMESTEPS="${NUM_TIMESTEPS:-100000000}"
NUM_ENVS="${NUM_ENVS:-8192}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
SEED="${SEED:-1}"
SUFFIX="${SUFFIX:-g1-rough-seed${SEED}}"
USE_TB="${USE_TB:-1}"
USE_WANDB="${USE_WANDB:-0}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Keep default compatible with legacy CUDA11 JAX stacks (e.g. 0.4.x on V100).
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"
if [ "${USE_CUDA:-1}" != "1" ]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
  export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
  echo "[train-rough] forcing CPU backend (USE_CUDA=${USE_CUDA:-0})"
fi

if [ "${USE_TB}" = "1" ]; then TB_FLAG="True"; else TB_FLAG="False"; fi
if [ "${USE_WANDB}" = "1" ]; then WANDB_FLAG="True"; else WANDB_FLAG="False"; fi

CMD=(
  python learning/train_jax_ppo.py
  --env_name=G1JoystickRoughTerrain
  --domain_randomization=True
  --num_timesteps="${NUM_TIMESTEPS}"
  --num_envs="${NUM_ENVS}"
  --num_eval_envs="${NUM_EVAL_ENVS}"
  --entropy_cost=0.005
  --clipping_epsilon=0.2
  --policy_hidden_layer_sizes=512,256,128
  --value_hidden_layer_sizes=512,256,128
  --policy_obs_key=state
  --value_obs_key=privileged_state
  --seed="${SEED}"
  --suffix="${SUFFIX}"
  --load_checkpoint_path="${FLAT_CKPT}"
  --use_tb="${TB_FLAG}"
  --use_wandb="${WANDB_FLAG}"
)

if [ -n "${PLAYGROUND_CONFIG_OVERRIDES:-}" ]; then
  CMD+=(--playground_config_overrides="${PLAYGROUND_CONFIG_OVERRIDES}")
fi

CKPT_ARG="${FLAT_CKPT}"
if [ -d "${FLAT_CKPT}" ]; then
  CKPT_BASE="$(basename "${FLAT_CKPT}")"
  HAS_CHILD_DIR=0
  if find "${FLAT_CKPT}" -mindepth 1 -maxdepth 1 -type d | read -r _; then
    HAS_CHILD_DIR=1
  fi
  if echo "${CKPT_BASE}" | grep -Eq '^[0-9]+$' && [ "${HAS_CHILD_DIR}" = "0" ]; then
    WRAP_DIR="${ROOT_DIR}/.resume_ckpt_rough"
    rm -rf "${WRAP_DIR}"
    mkdir -p "${WRAP_DIR}"
    if ln -s "${FLAT_CKPT}" "${WRAP_DIR}/${CKPT_BASE}" 2>/dev/null; then
      :
    else
      cp -a "${FLAT_CKPT}" "${WRAP_DIR}/${CKPT_BASE}"
    fi
    CKPT_ARG="${WRAP_DIR}"
    echo "[train-rough] wrapped checkpoint step dir ${FLAT_CKPT} as ${CKPT_ARG}/${CKPT_BASE}"
  fi
fi

for i in "${!CMD[@]}"; do
  if [ "${CMD[$i]}" = "--load_checkpoint_path=${FLAT_CKPT}" ]; then
    CMD[$i]="--load_checkpoint_path=${CKPT_ARG}"
  fi
done

echo "[train-rough] running: ${CMD[*]}"
"${CMD[@]}"
