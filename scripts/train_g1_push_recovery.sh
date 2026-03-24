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
  CUDA_NVCC_DIR="${VENV_SITE}/nvidia/cuda_nvcc"
  LIBDEVICE="${CUDA_NVCC_DIR}/nvvm/libdevice/libdevice.10.bc"
  if [ -d "${VENV_SITE}/nvidia" ]; then
    CUDA_LIB_DIRS="$(find "${VENV_SITE}/nvidia" -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')"
    if [ -n "${CUDA_LIB_DIRS}" ]; then
      export LD_LIBRARY_PATH="${CUDA_LIB_DIRS%:}:${LD_LIBRARY_PATH:-}"
      echo "[train-push] added CUDA libs from ${VENV_SITE}/nvidia to LD_LIBRARY_PATH"
    fi
  fi
  if [ -d "${CUDA_NVCC_DIR}" ]; then
    chmod -R a+rX "${CUDA_NVCC_DIR}" 2>/dev/null || true
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_NVCC_DIR} ${XLA_FLAGS:-}"
    echo "[train-push] set XLA cuda data dir: ${CUDA_NVCC_DIR}"
  fi
  if [ ! -r "${LIBDEVICE}" ]; then
    echo "ERROR: libdevice is not readable: ${LIBDEVICE}"
    ls -ld "${CUDA_NVCC_DIR}" "${CUDA_NVCC_DIR}/nvvm" "${CUDA_NVCC_DIR}/nvvm/libdevice" 2>/dev/null || true
    ls -l "${LIBDEVICE}" 2>/dev/null || true
    exit 1
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
ENTROPY_COST="${ENTROPY_COST:-0.005}"
USE_TB="${USE_TB:-1}"
USE_WANDB="${USE_WANDB:-0}"
MODE="${MODE:-ppo_dense}" # ppo_dense|maxrl_binary|maxrl_t
ADV_MODE="${ADV_MODE:-}"  # optional explicit override: ppo|maxrl_binary|maxrl_temporal
DRY_RUN="${DRY_RUN:-0}"
SCENARIO_GROUP_SIZE="${SCENARIO_GROUP_SIZE:-8}"
MAXRL_SCENARIO_KEY="${MAXRL_SCENARIO_KEY:-push_cfg}"
MAXRL_VERBOSE="${MAXRL_VERBOSE:-1}"
PUSH_ADV_MASK_MODE="${PUSH_ADV_MASK_MODE:-off}"   # off|post_push_soft|post_push_hard
PUSH_ADV_PRE_WEIGHT="${PUSH_ADV_PRE_WEIGHT:-0.1}" # used by post_push_soft
PUSH_MASK_SOURCE="${PUSH_MASK_SOURCE:-chunk}"     # chunk|stateful
PUSH_MASK_WINDOW_K="${PUSH_MASK_WINDOW_K:-20}"    # used by stateful source
PUSH_EVENT_EPS="${PUSH_EVENT_EPS:-1e-6}"
PUSH_ENTROPY_MODE="${PUSH_ENTROPY_MODE:-off}"     # off|post_push_additive
PUSH_ENTROPY_DELTA="${PUSH_ENTROPY_DELTA:-0.0}"
PUSH_REWARD_MODE="${PUSH_REWARD_MODE:-force_adaptive}"  # off|recovery_window|force_adaptive|recovery_gated
PUSH_REWARD_ABLATION_MODE="${PUSH_REWARD_ABLATION_MODE:-baseline}"  # baseline|survival_min|survival_stable|survival_phi
PUSH_INTERVAL_MIN="${PUSH_INTERVAL_MIN:-}"
PUSH_INTERVAL_MAX="${PUSH_INTERVAL_MAX:-}"
PUSH_MAGNITUDE_MIN="${PUSH_MAGNITUDE_MIN:-}"
PUSH_MAGNITUDE_MAX="${PUSH_MAGNITUDE_MAX:-}"
PUSH_DIRECTION_MODE="${PUSH_DIRECTION_MODE:-}"
PUSH_DIRECTION_FRAME="${PUSH_DIRECTION_FRAME:-}"
PUSH_BIASED_DIRECTION_PROB="${PUSH_BIASED_DIRECTION_PROB:-}"
PUSH_BIASED_DIRECTION_MIN_DEG="${PUSH_BIASED_DIRECTION_MIN_DEG:-}"
PUSH_BIASED_DIRECTION_MAX_DEG="${PUSH_BIASED_DIRECTION_MAX_DEG:-}"
PUSH_FIXED_ANGLE_DEG="${PUSH_FIXED_ANGLE_DEG:-}"
PUSH_SINGLE_PUSH="${PUSH_SINGLE_PUSH:-}"
SURVIVAL_ALIVE_SCALE="${SURVIVAL_ALIVE_SCALE:-1.0}"
SURVIVAL_TERMINATION_SCALE="${SURVIVAL_TERMINATION_SCALE:--100.0}"
SURVIVAL_ACTION_RATE_SCALE="${SURVIVAL_ACTION_RATE_SCALE:--0.0001}"
SURVIVAL_DOF_POS_LIMITS_SCALE="${SURVIVAL_DOF_POS_LIMITS_SCALE:--0.001}"
SURVIVAL_ANG_VEL_XY_SCALE="${SURVIVAL_ANG_VEL_XY_SCALE:--0.05}"
SURVIVAL_ORIENTATION_SCALE="${SURVIVAL_ORIENTATION_SCALE:--0.1}"
SURVIVAL_BASE_HEIGHT_SCALE="${SURVIVAL_BASE_HEIGHT_SCALE:--0.05}"
SURVIVAL_PHI_SCALE="${SURVIVAL_PHI_SCALE:-2.0}"
SURVIVAL_PHI_GAMMA="${SURVIVAL_PHI_GAMMA:-0.97}"
SURVIVAL_PHI_UPRIGHT_WEIGHT="${SURVIVAL_PHI_UPRIGHT_WEIGHT:-0.5}"
SURVIVAL_PHI_HEIGHT_WEIGHT="${SURVIVAL_PHI_HEIGHT_WEIGHT:-0.3}"
SURVIVAL_PHI_ANGVEL_WEIGHT="${SURVIVAL_PHI_ANGVEL_WEIGHT:-0.2}"
SURVIVAL_PHI_HEIGHT_MIN="${SURVIVAL_PHI_HEIGHT_MIN:-0.3}"
SURVIVAL_PHI_HEIGHT_NOMINAL="${SURVIVAL_PHI_HEIGHT_NOMINAL:-0.5}"
SURVIVAL_PHI_ANGVEL_K="${SURVIVAL_PHI_ANGVEL_K:-1.0}"
RECOVERY_WINDOW_K="${RECOVERY_WINDOW_K:-60}"
RECOVERY_WINDOW_TRACKING_SCALE="${RECOVERY_WINDOW_TRACKING_SCALE:-0.3}"
RECOVERY_WINDOW_TRACKING_SCALE_MIN="${RECOVERY_WINDOW_TRACKING_SCALE_MIN:-0.1}"
RECOVERY_OMEGA_WEIGHT="${RECOVERY_OMEGA_WEIGHT:-0.05}"
RECOVERY_ANG_MOM_WEIGHT="${RECOVERY_ANG_MOM_WEIGHT:-1.0}"
RECOVERY_ANG_MOM_SEVERITY_SCALE="${RECOVERY_ANG_MOM_SEVERITY_SCALE:-0.5}"
RECOVERY_ANG_MOM_SIGMA="${RECOVERY_ANG_MOM_SIGMA:-1.5}"
RECOVERY_UPRIGHT_WEIGHT="${RECOVERY_UPRIGHT_WEIGHT:-1.5}"
RECOVERY_UPRIGHT_SEVERITY_SCALE="${RECOVERY_UPRIGHT_SEVERITY_SCALE:-0.5}"
RECOVERY_UPRIGHT_SIGMA="${RECOVERY_UPRIGHT_SIGMA:-0.2}"
RECOVERY_COM_WEIGHT="${RECOVERY_COM_WEIGHT:-1.0}"
RECOVERY_COM_SEVERITY_SCALE="${RECOVERY_COM_SEVERITY_SCALE:-0.5}"
RECOVERY_COM_SIGMA="${RECOVERY_COM_SIGMA:-0.5}"
RECOVERY_STEP_WEIGHT="${RECOVERY_STEP_WEIGHT:-0.8}"
RECOVERY_STEP_AIR_TIME_MIN="${RECOVERY_STEP_AIR_TIME_MIN:-0.15}"
RECOVERY_SURVIVAL_WEIGHT="${RECOVERY_SURVIVAL_WEIGHT:-0.25}"
RECOVERY_BONUS="${RECOVERY_BONUS:-4.0}"
RECOVERY_BONUS_SEVERITY_SCALE="${RECOVERY_BONUS_SEVERITY_SCALE:-4.0}"
RECOVERY_BONUS_STABILITY_STEPS="${RECOVERY_BONUS_STABILITY_STEPS:-10}"
RECOVERY_BONUS_DELAY_STEPS="${RECOVERY_BONUS_DELAY_STEPS:-10}"
RECOVERY_STABLE_LIN_MIN="${RECOVERY_STABLE_LIN_MIN:-0.7}"
RECOVERY_STABLE_ANG_MIN="${RECOVERY_STABLE_ANG_MIN:-0.7}"
RECOVERY_GATED_ORIENTATION_SCALE="${RECOVERY_GATED_ORIENTATION_SCALE:-0.2}"
RECOVERY_GATED_BASE_HEIGHT_SCALE="${RECOVERY_GATED_BASE_HEIGHT_SCALE:-0.5}"
CAPTURE_POINT_LOG="${CAPTURE_POINT_LOG:-0}"

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

if { [ -n "${PUSH_INTERVAL_MIN}" ] && [ -z "${PUSH_INTERVAL_MAX}" ]; } || \
   { [ -z "${PUSH_INTERVAL_MIN}" ] && [ -n "${PUSH_INTERVAL_MAX}" ]; }; then
  echo "ERROR: set both PUSH_INTERVAL_MIN and PUSH_INTERVAL_MAX together."
  exit 1
fi

if { [ -n "${PUSH_MAGNITUDE_MIN}" ] && [ -z "${PUSH_MAGNITUDE_MAX}" ]; } || \
   { [ -z "${PUSH_MAGNITUDE_MIN}" ] && [ -n "${PUSH_MAGNITUDE_MAX}" ]; }; then
  echo "ERROR: set both PUSH_MAGNITUDE_MIN and PUSH_MAGNITUDE_MAX together."
  exit 1
fi

if { [ -n "${PUSH_BIASED_DIRECTION_MIN_DEG}" ] && [ -z "${PUSH_BIASED_DIRECTION_MAX_DEG}" ]; } || \
   { [ -z "${PUSH_BIASED_DIRECTION_MIN_DEG}" ] && [ -n "${PUSH_BIASED_DIRECTION_MAX_DEG}" ]; }; then
  echo "ERROR: set both PUSH_BIASED_DIRECTION_MIN_DEG and PUSH_BIASED_DIRECTION_MAX_DEG together."
  exit 1
fi

case "${PUSH_REWARD_ABLATION_MODE}" in
  baseline|survival_min|survival_stable|survival_phi) ;;
  *)
    echo "ERROR: invalid PUSH_REWARD_ABLATION_MODE=${PUSH_REWARD_ABLATION_MODE}. Expected baseline|survival_min|survival_stable|survival_phi"
    exit 1
    ;;
esac

if [ "${PUSH_REWARD_ABLATION_MODE}" != "baseline" ] && [ "${PUSH_REWARD_MODE}" != "off" ]; then
  echo "[train-push] forcing PUSH_REWARD_MODE=off because PUSH_REWARD_ABLATION_MODE=${PUSH_REWARD_ABLATION_MODE}"
  PUSH_REWARD_MODE="off"
fi

OVERRIDES_JSON="$("${VENV_DIR}/bin/python" -B - <<'PY'
import json
import os


def load_base_overrides():
  raw = os.environ.get("PLAYGROUND_CONFIG_OVERRIDES", "")
  if not raw:
    return {}
  return json.loads(raw)


def apply_recovery_reward_overrides(base):
  mode = os.environ["PUSH_REWARD_MODE"]
  if mode == "off":
    return
  base.setdefault("recovery_reward", {})
  base["recovery_reward"].update({
      "mode": mode,
      "window_steps": int(os.environ.get("RECOVERY_WINDOW_K", "60")),
      "tracking_scale": float(
          os.environ.get("RECOVERY_WINDOW_TRACKING_SCALE", "0.3")
      ),
      "tracking_scale_min": float(
          os.environ.get("RECOVERY_WINDOW_TRACKING_SCALE_MIN", "0.1")
      ),
      "omega_weight": float(os.environ.get("RECOVERY_OMEGA_WEIGHT", "0.05")),
      "ang_mom_weight": float(
          os.environ.get("RECOVERY_ANG_MOM_WEIGHT", "1.0")
      ),
      "ang_mom_severity_scale": float(
          os.environ.get("RECOVERY_ANG_MOM_SEVERITY_SCALE", "0.5")
      ),
      "ang_mom_sigma": float(os.environ.get("RECOVERY_ANG_MOM_SIGMA", "1.5")),
      "upright_weight": float(os.environ.get("RECOVERY_UPRIGHT_WEIGHT", "1.5")),
      "upright_severity_scale": float(
          os.environ.get("RECOVERY_UPRIGHT_SEVERITY_SCALE", "0.5")
      ),
      "upright_sigma": float(os.environ.get("RECOVERY_UPRIGHT_SIGMA", "0.2")),
      "com_weight": float(os.environ.get("RECOVERY_COM_WEIGHT", "1.0")),
      "com_severity_scale": float(
          os.environ.get("RECOVERY_COM_SEVERITY_SCALE", "0.5")
      ),
      "com_sigma": float(os.environ.get("RECOVERY_COM_SIGMA", "0.5")),
      "step_weight": float(os.environ.get("RECOVERY_STEP_WEIGHT", "0.8")),
      "step_air_time_min": float(
          os.environ.get("RECOVERY_STEP_AIR_TIME_MIN", "0.15")
      ),
      "survival_weight": float(
          os.environ.get("RECOVERY_SURVIVAL_WEIGHT", "0.25")
      ),
      "bonus": float(os.environ.get("RECOVERY_BONUS", "4.0")),
      "bonus_severity_scale": float(
          os.environ.get("RECOVERY_BONUS_SEVERITY_SCALE", "4.0")
      ),
      "bonus_stability_steps": int(
          os.environ.get("RECOVERY_BONUS_STABILITY_STEPS", "10")
      ),
      "bonus_delay_steps": int(
          os.environ.get("RECOVERY_BONUS_DELAY_STEPS", "10")
      ),
      "stable_lin_tracking_min": float(
          os.environ.get("RECOVERY_STABLE_LIN_MIN", "0.7")
      ),
      "stable_ang_tracking_min": float(
          os.environ.get("RECOVERY_STABLE_ANG_MIN", "0.7")
      ),
      "gated_orientation_scale": float(
          os.environ.get("RECOVERY_GATED_ORIENTATION_SCALE", "0.2")
      ),
      "gated_base_height_scale": float(
          os.environ.get("RECOVERY_GATED_BASE_HEIGHT_SCALE", "0.5")
      ),
      "capture_point_log": bool(int(os.environ.get("CAPTURE_POINT_LOG", "0"))),
  })


def parse_optional_bool(name):
  raw = os.environ.get(name, "").strip().lower()
  if not raw:
    return None
  if raw in {"1", "true", "yes", "on"}:
    return True
  if raw in {"0", "false", "no", "off"}:
    return False
  raise ValueError(f"invalid boolean for {name}: {raw}")


def apply_push_config_overrides(base):
  push_updates = {}

  interval_min = os.environ.get("PUSH_INTERVAL_MIN", "").strip()
  interval_max = os.environ.get("PUSH_INTERVAL_MAX", "").strip()
  if interval_min and interval_max:
    push_updates["interval_range"] = [float(interval_min), float(interval_max)]

  magnitude_min = os.environ.get("PUSH_MAGNITUDE_MIN", "").strip()
  magnitude_max = os.environ.get("PUSH_MAGNITUDE_MAX", "").strip()
  if magnitude_min and magnitude_max:
    push_updates["magnitude_range"] = [float(magnitude_min), float(magnitude_max)]

  direction_mode = os.environ.get("PUSH_DIRECTION_MODE", "").strip()
  if direction_mode:
    push_updates["direction_mode"] = direction_mode

  direction_frame = os.environ.get("PUSH_DIRECTION_FRAME", "").strip()
  if direction_frame:
    push_updates["direction_frame"] = direction_frame

  biased_direction_prob = os.environ.get(
      "PUSH_BIASED_DIRECTION_PROB", ""
  ).strip()
  biased_direction_min = os.environ.get(
      "PUSH_BIASED_DIRECTION_MIN_DEG", ""
  ).strip()
  biased_direction_max = os.environ.get(
      "PUSH_BIASED_DIRECTION_MAX_DEG", ""
  ).strip()
  if biased_direction_prob:
    push_updates["biased_direction_prob"] = float(biased_direction_prob)
  if biased_direction_min and biased_direction_max:
    push_updates["biased_direction_range_deg"] = [
        float(biased_direction_min),
        float(biased_direction_max),
    ]

  fixed_angle_deg = os.environ.get("PUSH_FIXED_ANGLE_DEG", "").strip()
  if fixed_angle_deg:
    push_updates["fixed_angle_deg"] = float(fixed_angle_deg)

  single_push = parse_optional_bool("PUSH_SINGLE_PUSH")
  if single_push is not None:
    push_updates["single_push"] = single_push

  if push_updates:
    base.setdefault("push_config", {})
    base["push_config"].update(push_updates)


def apply_reward_ablation_overrides(base):
  mode = os.environ["PUSH_REWARD_ABLATION_MODE"]
  if mode == "baseline":
    return

  zeroed_scales = {
      "tracking_lin_vel": 0.0,
      "tracking_ang_vel": 0.0,
      "recovery_ang_mom": 0.0,
      "recovery_bonus": 0.0,
      "recovery_upright": 0.0,
      "recovery_com_vel": 0.0,
      "recovery_step": 0.0,
      "recovery_survival": 0.0,
      "lin_vel_z": 0.0,
      "ang_vel_xy": 0.0,
      "orientation": 0.0,
      "base_height": 0.0,
      "torques": 0.0,
      "action_rate": 0.0,
      "energy": 0.0,
      "dof_acc": 0.0,
      "feet_clearance": 0.0,
      "feet_air_time": 0.0,
      "feet_slip": 0.0,
      "feet_height": 0.0,
      "feet_phase": 0.0,
      "alive": 0.0,
      "survival_phi": 0.0,
      "stand_still": 0.0,
      "termination": 0.0,
      "collision": 0.0,
      "contact_force": 0.0,
      "joint_deviation_knee": 0.0,
      "joint_deviation_hip": 0.0,
      "dof_pos_limits": 0.0,
      "pose": 0.0,
  }
  zeroed_scales.update({
      "alive": float(os.environ.get("SURVIVAL_ALIVE_SCALE", "1.0")),
      "termination": float(os.environ.get("SURVIVAL_TERMINATION_SCALE", "-100.0")),
      "action_rate": float(os.environ.get("SURVIVAL_ACTION_RATE_SCALE", "-0.0001")),
      "dof_pos_limits": float(os.environ.get("SURVIVAL_DOF_POS_LIMITS_SCALE", "-0.001")),
  })
  if mode == "survival_stable":
    zeroed_scales.update({
        "ang_vel_xy": float(os.environ.get("SURVIVAL_ANG_VEL_XY_SCALE", "-0.05")),
        "orientation": float(os.environ.get("SURVIVAL_ORIENTATION_SCALE", "-0.1")),
        "base_height": float(os.environ.get("SURVIVAL_BASE_HEIGHT_SCALE", "-0.05")),
    })
  elif mode == "survival_phi":
    zeroed_scales.update({
        "survival_phi": float(os.environ.get("SURVIVAL_PHI_SCALE", "2.0")),
    })

  base.setdefault("reward_config", {})
  base["reward_config"].setdefault("scales", {})
  base["reward_config"]["scales"].update(zeroed_scales)
  base.setdefault("recovery_reward", {})
  if mode == "survival_phi":
    base["recovery_reward"].update({
        "mode": "survival_phi",
        "survival_phi_gamma": float(os.environ.get("SURVIVAL_PHI_GAMMA", "0.97")),
        "survival_phi_upright_weight": float(os.environ.get("SURVIVAL_PHI_UPRIGHT_WEIGHT", "0.5")),
        "survival_phi_height_weight": float(os.environ.get("SURVIVAL_PHI_HEIGHT_WEIGHT", "0.3")),
        "survival_phi_angvel_weight": float(os.environ.get("SURVIVAL_PHI_ANGVEL_WEIGHT", "0.2")),
        "survival_phi_height_min": float(os.environ.get("SURVIVAL_PHI_HEIGHT_MIN", "0.3")),
        "survival_phi_height_nominal": float(os.environ.get("SURVIVAL_PHI_HEIGHT_NOMINAL", "0.5")),
        "survival_phi_angvel_k": float(os.environ.get("SURVIVAL_PHI_ANGVEL_K", "1.0")),
    })
  else:
    base["recovery_reward"]["mode"] = "off"


base = load_base_overrides()
apply_recovery_reward_overrides(base)
apply_push_config_overrides(base)
apply_reward_ablation_overrides(base)
print(json.dumps(base, separators=(",", ":")))
PY
)"

if [ -n "${OVERRIDES_JSON}" ] && [ "${OVERRIDES_JSON}" != "{}" ]; then
  PLAYGROUND_CONFIG_OVERRIDES="${OVERRIDES_JSON}"
fi

CMD=(
  python learning/train_jax_ppo.py
  --env_name="${ENV_NAME}"
  --domain_randomization=True
  --num_timesteps="${NUM_TIMESTEPS}"
  --num_envs="${NUM_ENVS}"
  --num_eval_envs="${NUM_EVAL_ENVS}"
  --entropy_cost="${ENTROPY_COST}"
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
  --push_mask_source="${PUSH_MASK_SOURCE}"
  --push_mask_window_k="${PUSH_MASK_WINDOW_K}"
  --push_event_eps="${PUSH_EVENT_EPS}"
  --push_entropy_mode="${PUSH_ENTROPY_MODE}"
  --push_entropy_delta="${PUSH_ENTROPY_DELTA}"
  --use_tb="${TB_FLAG}"
  --use_wandb="${WANDB_FLAG}"
)

if [ -n "${PLAYGROUND_CONFIG_OVERRIDES:-}" ]; then
  CMD+=(--playground_config_overrides="${PLAYGROUND_CONFIG_OVERRIDES}")
fi

RESOLVED_ROUGH_CKPT="$("${RESOLVE_CKPT_SCRIPT}" "${ROUGH_CKPT}")"
echo "[train-push] resolved rough checkpoint: ${RESOLVED_ROUGH_CKPT}"

CKPT_ARG="${RESOLVED_ROUGH_CKPT}"
if [ -d "${RESOLVED_ROUGH_CKPT}" ]; then
  CKPT_BASE="$(basename "${RESOLVED_ROUGH_CKPT}")"
  if echo "${CKPT_BASE}" | grep -Eq '^[0-9]+$'; then
    WRAP_DIR="${ROOT_DIR}/.resume_ckpt_push"
    rm -rf "${WRAP_DIR}"
    mkdir -p "${WRAP_DIR}"
    if ln -s "${RESOLVED_ROUGH_CKPT}" "${WRAP_DIR}/${CKPT_BASE}" 2>/dev/null; then
      :
    else
      cp -a "${RESOLVED_ROUGH_CKPT}" "${WRAP_DIR}/${CKPT_BASE}"
    fi
    CKPT_ARG="${WRAP_DIR}"
    echo "[train-push] wrapped checkpoint step dir ${RESOLVED_ROUGH_CKPT} as ${CKPT_ARG}/${CKPT_BASE}"
  fi
fi
CMD+=(--load_checkpoint_path="${CKPT_ARG}")

echo "[train-push] mode=${MODE} adv_mode=${ADV_MODE_FLAG} scenario_group_size=${SCENARIO_GROUP_SIZE} push_adv_mask_mode=${PUSH_ADV_MASK_MODE} push_mask_source=${PUSH_MASK_SOURCE} push_mask_window_k=${PUSH_MASK_WINDOW_K} push_entropy_mode=${PUSH_ENTROPY_MODE} push_reward_mode=${PUSH_REWARD_MODE} reward_ablation_mode=${PUSH_REWARD_ABLATION_MODE}"
if [ -n "${PLAYGROUND_CONFIG_OVERRIDES:-}" ]; then
  echo "[train-push] playground_config_overrides=${PLAYGROUND_CONFIG_OVERRIDES}"
fi
echo "[train-push] running: ${CMD[*]}"
if [ "${DRY_RUN}" = "1" ]; then
  exit 0
fi
"${CMD[@]}"
