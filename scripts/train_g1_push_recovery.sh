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
if [ -z "${ROUGH_CKPT:-}" ]; then
  echo "ERROR: ROUGH_CKPT is not set."
  echo "Set it to a rough run checkpoint directory or step directory."
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
      echo "[train-push] added CUDA libs from ${VENV_SITE}/nvidia to LD_LIBRARY_PATH"
    fi
  fi
fi

cd "${PLAYGROUND_DIR}"
mkdir -p logs
chmod 777 logs || true

ENV_NAME="${ENV_NAME:-G1JoystickRoughTerrain}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50000000}"
NUM_ENVS="${NUM_ENVS:-1024}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
SEED="${SEED:-1}"
SUFFIX="${SUFFIX:-g1-push-recovery-seed${SEED}}"
USE_TB="${USE_TB:-1}"
USE_WANDB="${USE_WANDB:-0}"
MODE="${MODE:-ppo_dense}" # ppo_dense|maxrl_binary|maxrl_t
ADV_MODE="${ADV_MODE:-}"  # optional explicit override: ppo|maxrl_binary|maxrl_temporal
SCENARIO_GROUP_SIZE="${SCENARIO_GROUP_SIZE:-8}"
MAXRL_SCENARIO_KEY="${MAXRL_SCENARIO_KEY:-push_cfg}"
MAXRL_VERBOSE="${MAXRL_VERBOSE:-1}"
PUSH_ADV_MASK_MODE="${PUSH_ADV_MASK_MODE:-off}"   # off|post_push_soft|post_push_hard
PUSH_ADV_PRE_WEIGHT="${PUSH_ADV_PRE_WEIGHT:-0.1}" # used by post_push_soft
PUSH_EVENT_EPS="${PUSH_EVENT_EPS:-1e-6}"
PUSH_ENTROPY_MODE="${PUSH_ENTROPY_MODE:-off}"     # off|post_push_additive
PUSH_ENTROPY_DELTA="${PUSH_ENTROPY_DELTA:-0.0}"

case "${MODE}" in
  ppo_dense)
    ADV_MODE_FLAG="ppo"
    MAXRL_LOG_ONLY_FLAG="${MAXRL_LOG_ONLY:-0}"
    ;;
  maxrl_binary)
    ADV_MODE_FLAG="maxrl_binary"
    MAXRL_LOG_ONLY_FLAG="${MAXRL_LOG_ONLY:-0}"
    ;;
  maxrl_t)
    ADV_MODE_FLAG="maxrl_temporal"
    MAXRL_LOG_ONLY_FLAG="${MAXRL_LOG_ONLY:-0}"
    ;;
  *)
    echo "ERROR: invalid MODE=${MODE}. Expected ppo_dense|maxrl_binary|maxrl_t"
    exit 1
    ;;
esac

if [ -n "${ADV_MODE}" ]; then
  case "${ADV_MODE}" in
    ppo|maxrl_binary|maxrl_temporal)
      ADV_MODE_FLAG="${ADV_MODE}"
      ;;
    *)
      echo "ERROR: invalid ADV_MODE=${ADV_MODE}. Expected ppo|maxrl_binary|maxrl_temporal"
      exit 1
      ;;
  esac
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"
if [ "${USE_CUDA:-1}" != "1" ]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
  export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
  echo "[train-push] forcing CPU backend (USE_CUDA=${USE_CUDA:-0})"
fi

if [ "${USE_TB}" = "1" ]; then TB_FLAG="True"; else TB_FLAG="False"; fi
if [ "${USE_WANDB}" = "1" ]; then WANDB_FLAG="True"; else WANDB_FLAG="False"; fi
if [ "${MAXRL_LOG_ONLY_FLAG}" = "1" ]; then MAXRL_LOG_ONLY_BOOL="True"; else MAXRL_LOG_ONLY_BOOL="False"; fi
if [ "${MAXRL_VERBOSE}" = "1" ]; then MAXRL_VERBOSE_BOOL="True"; else MAXRL_VERBOSE_BOOL="False"; fi

CMD=(
  python learning/train_jax_ppo.py
  --env_name="${ENV_NAME}"
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
  --adv_mode="${ADV_MODE_FLAG}"
  --scenario_group_size="${SCENARIO_GROUP_SIZE}"
  --maxrl_log_only="${MAXRL_LOG_ONLY_BOOL}"
  --maxrl_scenario_key="${MAXRL_SCENARIO_KEY}"
  --maxrl_verbose="${MAXRL_VERBOSE_BOOL}"
  --push_adv_mask_mode="${PUSH_ADV_MASK_MODE}"
  --push_adv_pre_weight="${PUSH_ADV_PRE_WEIGHT}"
  --push_event_eps="${PUSH_EVENT_EPS}"
  --push_entropy_mode="${PUSH_ENTROPY_MODE}"
  --push_entropy_delta="${PUSH_ENTROPY_DELTA}"
  --use_tb="${TB_FLAG}"
  --use_wandb="${WANDB_FLAG}"
)

if [ -n "${PLAYGROUND_CONFIG_OVERRIDES:-}" ]; then
  CMD+=(--playground_config_overrides="${PLAYGROUND_CONFIG_OVERRIDES}")
fi

CKPT_ARG="${ROUGH_CKPT}"
if [ -d "${ROUGH_CKPT}" ]; then
  CKPT_BASE="$(basename "${ROUGH_CKPT}")"
  HAS_CHILD_DIR=0
  if find "${ROUGH_CKPT}" -mindepth 1 -maxdepth 1 -type d | read -r _; then
    HAS_CHILD_DIR=1
  fi
  if echo "${CKPT_BASE}" | grep -Eq '^[0-9]+$' && [ "${HAS_CHILD_DIR}" = "0" ]; then
    WRAP_DIR="${ROOT_DIR}/.resume_ckpt_push"
    rm -rf "${WRAP_DIR}"
    mkdir -p "${WRAP_DIR}"
    if ln -s "${ROUGH_CKPT}" "${WRAP_DIR}/${CKPT_BASE}" 2>/dev/null; then
      :
    else
      cp -a "${ROUGH_CKPT}" "${WRAP_DIR}/${CKPT_BASE}"
    fi
    CKPT_ARG="${WRAP_DIR}"
    echo "[train-push] wrapped checkpoint step dir ${ROUGH_CKPT} as ${CKPT_ARG}/${CKPT_BASE}"
  fi
fi
CMD+=(--load_checkpoint_path="${CKPT_ARG}")

echo "[train-push] mode=${MODE} adv_mode=${ADV_MODE_FLAG} scenario_group_size=${SCENARIO_GROUP_SIZE} push_adv_mask_mode=${PUSH_ADV_MASK_MODE} push_entropy_mode=${PUSH_ENTROPY_MODE}"
echo "[train-push] running: ${CMD[*]}"
"${CMD[@]}"
