#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PLAYGROUND_DIR="${ROOT_DIR}/mujoco_playground"
RESOLVE_CKPT_SCRIPT="${ROOT_DIR}/scripts/resolve_ppo_checkpoint.sh"

if [ ! -d "${VENV_DIR}" ]; then
  echo "ERROR: missing ${VENV_DIR}. Run: bash scripts/bootstrap_env.sh"
  exit 1
fi
if [ ! -d "${PLAYGROUND_DIR}" ]; then
  echo "ERROR: missing ${PLAYGROUND_DIR}. Run: bash scripts/bootstrap_env.sh"
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
      echo "[train-flat] added CUDA libs from ${VENV_SITE}/nvidia to LD_LIBRARY_PATH"
    fi
  fi
fi
cd "${PLAYGROUND_DIR}"
mkdir -p logs
chmod 777 logs || true

NUM_TIMESTEPS="${NUM_TIMESTEPS:-200000000}"
NUM_ENVS="${NUM_ENVS:-8192}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
SEED="${SEED:-1}"
SUFFIX="${SUFFIX:-g1-flat-seed${SEED}}"
USE_TB="${USE_TB:-1}"
USE_WANDB="${USE_WANDB:-0}"
ADV_MODE="${ADV_MODE:-}"
SCENARIO_GROUP_SIZE="${SCENARIO_GROUP_SIZE:-}"
MAXRL_LOG_ONLY="${MAXRL_LOG_ONLY:-}"
MAXRL_SCENARIO_KEY="${MAXRL_SCENARIO_KEY:-}"
MAXRL_VERBOSE="${MAXRL_VERBOSE:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Keep default compatible with legacy CUDA11 JAX stacks (e.g. 0.4.x on V100).
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"
if [ "${USE_CUDA:-1}" != "1" ]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
  export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
  echo "[train-flat] forcing CPU backend (USE_CUDA=${USE_CUDA:-0})"
fi

if [ "${USE_TB}" = "1" ]; then TB_FLAG="True"; else TB_FLAG="False"; fi
if [ "${USE_WANDB}" = "1" ]; then WANDB_FLAG="True"; else WANDB_FLAG="False"; fi

CMD=(
  python learning/train_jax_ppo.py
  --env_name=G1JoystickFlatTerrain
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
  --use_tb="${TB_FLAG}"
  --use_wandb="${WANDB_FLAG}"
)

if [ -n "${PLAYGROUND_CONFIG_OVERRIDES:-}" ]; then
  CMD+=(--playground_config_overrides="${PLAYGROUND_CONFIG_OVERRIDES}")
fi
if [ -n "${ADV_MODE}" ]; then
  CMD+=(--adv_mode="${ADV_MODE}")
fi
if [ -n "${SCENARIO_GROUP_SIZE}" ]; then
  CMD+=(--scenario_group_size="${SCENARIO_GROUP_SIZE}")
fi
if [ -n "${MAXRL_LOG_ONLY}" ]; then
  CMD+=(--maxrl_log_only="${MAXRL_LOG_ONLY}")
fi
if [ -n "${MAXRL_SCENARIO_KEY}" ]; then
  CMD+=(--maxrl_scenario_key="${MAXRL_SCENARIO_KEY}")
fi
if [ -n "${MAXRL_VERBOSE}" ]; then
  CMD+=(--maxrl_verbose="${MAXRL_VERBOSE}")
fi
if [ -n "${LOAD_CKPT:-}" ]; then
  RESOLVED_LOAD_CKPT="$("${RESOLVE_CKPT_SCRIPT}" "${LOAD_CKPT}")"
  echo "[train-flat] resolved checkpoint: ${RESOLVED_LOAD_CKPT}"
  CKPT_ARG="${RESOLVED_LOAD_CKPT}"
  if [ -d "${RESOLVED_LOAD_CKPT}" ]; then
    CKPT_BASE="$(basename "${RESOLVED_LOAD_CKPT}")"
    if echo "${CKPT_BASE}" | grep -Eq '^[0-9]+$'; then
      # train_jax_ppo expects a directory whose children are step directories.
      WRAP_DIR="${ROOT_DIR}/.resume_ckpt_flat"
      rm -rf "${WRAP_DIR}"
      mkdir -p "${WRAP_DIR}"
      if ln -s "${RESOLVED_LOAD_CKPT}" "${WRAP_DIR}/${CKPT_BASE}" 2>/dev/null; then
        :
      else
        cp -a "${RESOLVED_LOAD_CKPT}" "${WRAP_DIR}/${CKPT_BASE}"
      fi
      CKPT_ARG="${WRAP_DIR}"
      echo "[train-flat] wrapped checkpoint step dir ${RESOLVED_LOAD_CKPT} as ${CKPT_ARG}/${CKPT_BASE}"
    fi
  fi
  CMD+=(--load_checkpoint_path="${CKPT_ARG}")
fi

echo "[train-flat] running: ${CMD[*]}"
"${CMD[@]}"
