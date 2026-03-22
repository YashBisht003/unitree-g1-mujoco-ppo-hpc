#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PLAYGROUND_DIR="${ROOT_DIR}/mujoco_playground"
LOG_ROOT="${PLAYGROUND_DIR}/logs"
RESULTS_DIR="${ROOT_DIR}/research/results"
RESOLVE_CKPT_SCRIPT="${ROOT_DIR}/scripts/resolve_ppo_checkpoint.sh"

if [ ! -d "${VENV_DIR}" ]; then
  echo "ERROR: missing ${VENV_DIR}. Run: bash scripts/bootstrap_env.sh"
  exit 1
fi
if [ ! -d "${PLAYGROUND_DIR}" ]; then
  echo "ERROR: missing ${PLAYGROUND_DIR}. Run: bash scripts/bootstrap_env.sh"
  exit 1
fi

PROFILE="${PROFILE:-single_gpu}"         # baseline|single_gpu|smoke
START_STAGE="${START_STAGE:-flat}"       # flat|rough|push|eval
PIPELINE_ID="${PIPELINE_ID:-$(date +%Y%m%d-%H%M%S)}"
SEED="${SEED:-1}"
USE_CUDA="${USE_CUDA:-1}"
USE_TB="${USE_TB:-1}"
USE_WANDB="${USE_WANDB:-0}"
RUN_PUSH_EVAL="${RUN_PUSH_EVAL:-1}"
PUSH_REWARD_MODE="${PUSH_REWARD_MODE:-force_adaptive}"
DRY_RUN="${DRY_RUN:-0}"

PUSH_EVAL_MAGNITUDES="${PUSH_EVAL_MAGNITUDES:-1,1.5,2,2.25,2.5,2.75,3}"
PUSH_EVAL_EPISODES="${PUSH_EVAL_EPISODES:-300}"
PUSH_EVAL_BATCH_SIZE="${PUSH_EVAL_BATCH_SIZE:-32}"
PUSH_EVAL_EPISODE_LENGTH="${PUSH_EVAL_EPISODE_LENGTH:-1000}"
PUSH_EVAL_RECOVERY_WINDOW_S="${PUSH_EVAL_RECOVERY_WINDOW_S:-5.0}"
PUSH_EVAL_INTERVAL_S="${PUSH_EVAL_INTERVAL_S:-2.0}"
PUSH_EVAL_DETERMINISTIC="${PUSH_EVAL_DETERMINISTIC:-1}"

case "${PROFILE}" in
  baseline)
    : "${FLAT_NUM_TIMESTEPS:=200000000}"
    : "${FLAT_NUM_ENVS:=8192}"
    : "${FLAT_NUM_EVAL_ENVS:=128}"
    : "${ROUGH_NUM_TIMESTEPS:=100000000}"
    : "${ROUGH_NUM_ENVS:=8192}"
    : "${ROUGH_NUM_EVAL_ENVS:=128}"
    : "${PUSH_NUM_TIMESTEPS:=50000000}"
    : "${PUSH_NUM_ENVS:=1024}"
    : "${PUSH_NUM_EVAL_ENVS:=128}"
    ;;
  single_gpu)
    : "${FLAT_NUM_TIMESTEPS:=100000000}"
    : "${FLAT_NUM_ENVS:=4096}"
    : "${FLAT_NUM_EVAL_ENVS:=128}"
    : "${ROUGH_NUM_TIMESTEPS:=50000000}"
    : "${ROUGH_NUM_ENVS:=4096}"
    : "${ROUGH_NUM_EVAL_ENVS:=128}"
    : "${PUSH_NUM_TIMESTEPS:=25000000}"
    : "${PUSH_NUM_ENVS:=1024}"
    : "${PUSH_NUM_EVAL_ENVS:=128}"
    ;;
  smoke)
    : "${FLAT_NUM_TIMESTEPS:=1310720}"
    : "${FLAT_NUM_ENVS:=512}"
    : "${FLAT_NUM_EVAL_ENVS:=64}"
    : "${ROUGH_NUM_TIMESTEPS:=1310720}"
    : "${ROUGH_NUM_ENVS:=512}"
    : "${ROUGH_NUM_EVAL_ENVS:=64}"
    : "${PUSH_NUM_TIMESTEPS:=1310720}"
    : "${PUSH_NUM_ENVS:=256}"
    : "${PUSH_NUM_EVAL_ENVS:=64}"
    : "${PUSH_EVAL_EPISODES:=50}"
    ;;
  *)
    echo "ERROR: invalid PROFILE=${PROFILE}. Expected baseline|single_gpu|smoke"
    exit 1
    ;;
esac

case "${START_STAGE}" in
  flat|rough|push|eval) ;;
  *)
    echo "ERROR: invalid START_STAGE=${START_STAGE}. Expected flat|rough|push|eval"
    exit 1
    ;;
esac

FLAT_SUFFIX="${FLAT_SUFFIX:-g1-flat-${PROFILE}-seed${SEED}-${PIPELINE_ID}}"
ROUGH_SUFFIX="${ROUGH_SUFFIX:-g1-rough-${PROFILE}-seed${SEED}-${PIPELINE_ID}}"
PUSH_SUFFIX="${PUSH_SUFFIX:-g1-push-${PROFILE}-${PUSH_REWARD_MODE}-seed${SEED}-${PIPELINE_ID}}"
FLAT_LOAD_CKPT="${FLAT_LOAD_CKPT:-}"
EVAL_BASENAME="${EVAL_BASENAME:-recovery_rate_${PUSH_SUFFIX}}"
EVAL_JSON="${EVAL_JSON:-${RESULTS_DIR}/${EVAL_BASENAME}.json}"
EVAL_CSV="${EVAL_CSV:-${RESULTS_DIR}/${EVAL_BASENAME}.csv}"

mkdir -p "${LOG_ROOT}" "${RESULTS_DIR}"

stage_rank() {
  case "$1" in
    flat) echo 1 ;;
    rough) echo 2 ;;
    push) echo 3 ;;
    eval) echo 4 ;;
  esac
}

run_env_script() {
  local label="$1"
  local script_path="$2"
  shift 2
  echo "[pipeline] ${label}: $(basename "${script_path}")"
  if [ "${DRY_RUN}" = "1" ]; then
    printf '[pipeline] dry-run: env'
    for kv in "$@"; do
      printf ' %q' "${kv}"
    done
    printf ' bash %q\n' "${script_path}"
    return 0
  fi
  env "$@" bash "${script_path}"
}

resolve_valid_ckpt() {
  local ckpt_path="$1"
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%s\n' "${ckpt_path}"
    return 0
  fi
  bash "${RESOLVE_CKPT_SCRIPT}" "${ckpt_path}"
}

latest_step_dir_from_run() {
  local run_dir="$1"
  local ckpt_dir="${run_dir}/checkpoints"
  local latest_step
  if [ ! -d "${ckpt_dir}" ]; then
    return 1
  fi
  latest_step="$(find "${ckpt_dir}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -n 1)"
  if [ -z "${latest_step}" ]; then
    return 1
  fi
  printf '%s\n' "${ckpt_dir}/${latest_step}"
}

latest_step_dir_for_suffix() {
  local env_prefix="$1"
  local suffix="$2"
  local run_dir
  run_dir="$(find "${LOG_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "${env_prefix}-*-${suffix}" | sort -r | head -n 1)"
  if [ -z "${run_dir}" ]; then
    return 1
  fi
  latest_step_dir_from_run "${run_dir}"
}

resolve_flat_ckpt() {
  if [ -n "${FLAT_CKPT:-}" ]; then
    printf '%s\n' "${FLAT_CKPT}"
    return 0
  fi
  latest_step_dir_for_suffix "G1JoystickFlatTerrain" "${FLAT_SUFFIX}"
}

resolve_rough_ckpt() {
  if [ -n "${ROUGH_CKPT:-}" ]; then
    printf '%s\n' "${ROUGH_CKPT}"
    return 0
  fi
  latest_step_dir_for_suffix "G1JoystickRoughTerrain" "${ROUGH_SUFFIX}"
}

resolve_push_ckpt() {
  if [ -n "${PUSH_CKPT:-}" ]; then
    printf '%s\n' "${PUSH_CKPT}"
    return 0
  fi
  latest_step_dir_for_suffix "G1JoystickRoughTerrain" "${PUSH_SUFFIX}"
}

placeholder_ckpt() {
  local label="$1"
  printf '%s\n' "${LOG_ROOT}/${label}/checkpoints/000000000000"
}

echo "[pipeline] profile=${PROFILE} start_stage=${START_STAGE} seed=${SEED} pipeline_id=${PIPELINE_ID}"
echo "[pipeline] flat:  envs=${FLAT_NUM_ENVS} eval_envs=${FLAT_NUM_EVAL_ENVS} timesteps=${FLAT_NUM_TIMESTEPS} suffix=${FLAT_SUFFIX}"
if [ -n "${FLAT_LOAD_CKPT}" ]; then
  echo "[pipeline] flat resume checkpoint: ${FLAT_LOAD_CKPT}"
fi
echo "[pipeline] rough: envs=${ROUGH_NUM_ENVS} eval_envs=${ROUGH_NUM_EVAL_ENVS} timesteps=${ROUGH_NUM_TIMESTEPS} suffix=${ROUGH_SUFFIX}"
echo "[pipeline] push:  envs=${PUSH_NUM_ENVS} eval_envs=${PUSH_NUM_EVAL_ENVS} timesteps=${PUSH_NUM_TIMESTEPS} suffix=${PUSH_SUFFIX} reward_mode=${PUSH_REWARD_MODE}"

if [ "$(stage_rank "${START_STAGE}")" -le 1 ]; then
  run_env_script "stage=flat" "${ROOT_DIR}/scripts/train_g1_flat.sh" \
    "USE_CUDA=${USE_CUDA}" \
    "USE_TB=${USE_TB}" \
    "USE_WANDB=${USE_WANDB}" \
    "SEED=${SEED}" \
    "SUFFIX=${FLAT_SUFFIX}" \
    "LOAD_CKPT=${FLAT_LOAD_CKPT}" \
    "NUM_TIMESTEPS=${FLAT_NUM_TIMESTEPS}" \
    "NUM_ENVS=${FLAT_NUM_ENVS}" \
    "NUM_EVAL_ENVS=${FLAT_NUM_EVAL_ENVS}" \
    "PLAYGROUND_CONFIG_OVERRIDES=${FLAT_PLAYGROUND_CONFIG_OVERRIDES:-}"
fi

if [ "$(stage_rank "${START_STAGE}")" -le 2 ]; then
  FLAT_CKPT_RESOLVED="$(resolve_flat_ckpt || true)"
  if [ -z "${FLAT_CKPT_RESOLVED:-}" ]; then
    if [ "${DRY_RUN}" = "1" ]; then
      FLAT_CKPT_RESOLVED="$(placeholder_ckpt "G1JoystickFlatTerrain-YYYYMMDD-HHMMSS-${FLAT_SUFFIX}")"
    else
      echo "ERROR: could not resolve FLAT_CKPT after flat stage."
      exit 1
    fi
  fi
  FLAT_CKPT_RESOLVED="$(resolve_valid_ckpt "${FLAT_CKPT_RESOLVED}")"
  echo "[pipeline] resolved flat checkpoint: ${FLAT_CKPT_RESOLVED}"
  run_env_script "stage=rough" "${ROOT_DIR}/scripts/train_g1_rough_from_flat.sh" \
    "USE_CUDA=${USE_CUDA}" \
    "USE_TB=${USE_TB}" \
    "USE_WANDB=${USE_WANDB}" \
    "SEED=${SEED}" \
    "SUFFIX=${ROUGH_SUFFIX}" \
    "FLAT_CKPT=${FLAT_CKPT_RESOLVED}" \
    "NUM_TIMESTEPS=${ROUGH_NUM_TIMESTEPS}" \
    "NUM_ENVS=${ROUGH_NUM_ENVS}" \
    "NUM_EVAL_ENVS=${ROUGH_NUM_EVAL_ENVS}" \
    "PLAYGROUND_CONFIG_OVERRIDES=${ROUGH_PLAYGROUND_CONFIG_OVERRIDES:-}"
fi

if [ "$(stage_rank "${START_STAGE}")" -le 3 ]; then
  ROUGH_CKPT_RESOLVED="$(resolve_rough_ckpt || true)"
  if [ -z "${ROUGH_CKPT_RESOLVED:-}" ]; then
    if [ "${DRY_RUN}" = "1" ]; then
      ROUGH_CKPT_RESOLVED="$(placeholder_ckpt "G1JoystickRoughTerrain-YYYYMMDD-HHMMSS-${ROUGH_SUFFIX}")"
    else
      echo "ERROR: could not resolve ROUGH_CKPT after rough stage."
      exit 1
    fi
  fi
  ROUGH_CKPT_RESOLVED="$(resolve_valid_ckpt "${ROUGH_CKPT_RESOLVED}")"
  echo "[pipeline] resolved rough checkpoint: ${ROUGH_CKPT_RESOLVED}"
  run_env_script "stage=push" "${ROOT_DIR}/scripts/train_g1_push_recovery.sh" \
    "USE_CUDA=${USE_CUDA}" \
    "USE_TB=${USE_TB}" \
    "USE_WANDB=${USE_WANDB}" \
    "SEED=${SEED}" \
    "SUFFIX=${PUSH_SUFFIX}" \
    "ROUGH_CKPT=${ROUGH_CKPT_RESOLVED}" \
    "PUSH_REWARD_MODE=${PUSH_REWARD_MODE}" \
    "NUM_TIMESTEPS=${PUSH_NUM_TIMESTEPS}" \
    "NUM_ENVS=${PUSH_NUM_ENVS}" \
    "NUM_EVAL_ENVS=${PUSH_NUM_EVAL_ENVS}" \
    "PLAYGROUND_CONFIG_OVERRIDES=${PUSH_PLAYGROUND_CONFIG_OVERRIDES:-}"
fi

if [ "${RUN_PUSH_EVAL}" = "1" ] && [ "$(stage_rank "${START_STAGE}")" -le 4 ]; then
  PUSH_CKPT_RESOLVED="$(resolve_push_ckpt || true)"
  if [ -z "${PUSH_CKPT_RESOLVED:-}" ]; then
    if [ "${DRY_RUN}" = "1" ]; then
      PUSH_CKPT_RESOLVED="$(placeholder_ckpt "G1JoystickRoughTerrain-YYYYMMDD-HHMMSS-${PUSH_SUFFIX}")"
    else
      echo "ERROR: could not resolve PUSH_CKPT for evaluation."
      exit 1
    fi
  fi
  PUSH_CKPT_RESOLVED="$(resolve_valid_ckpt "${PUSH_CKPT_RESOLVED}")"
  echo "[pipeline] resolved push checkpoint: ${PUSH_CKPT_RESOLVED}"
  if [ "${DRY_RUN}" = "1" ]; then
    printf '[pipeline] dry-run: %q %q %q' "${VENV_DIR}/bin/python" "${ROOT_DIR}/research/eval_recovery_rate.py" "--checkpoint_path=${PUSH_CKPT_RESOLVED}"
    printf ' %q' \
      "--episodes_per_magnitude=${PUSH_EVAL_EPISODES}" \
      "--batch_size=${PUSH_EVAL_BATCH_SIZE}" \
      "--episode_length=${PUSH_EVAL_EPISODE_LENGTH}" \
      "--recovery_window_s=${PUSH_EVAL_RECOVERY_WINDOW_S}" \
      "--push_interval_s=${PUSH_EVAL_INTERVAL_S}" \
      "--push_magnitudes=${PUSH_EVAL_MAGNITUDES}" \
      "--output_json=${EVAL_JSON}" \
      "--output_csv=${EVAL_CSV}"
    if [ "${PUSH_EVAL_DETERMINISTIC}" = "1" ]; then
      printf ' %q' "--deterministic"
    fi
    printf '\n'
  else
    EVAL_CMD=(
      "${VENV_DIR}/bin/python"
      "${ROOT_DIR}/research/eval_recovery_rate.py"
      --checkpoint_path="${PUSH_CKPT_RESOLVED}"
      --episodes_per_magnitude="${PUSH_EVAL_EPISODES}"
      --batch_size="${PUSH_EVAL_BATCH_SIZE}"
      --episode_length="${PUSH_EVAL_EPISODE_LENGTH}"
      --recovery_window_s="${PUSH_EVAL_RECOVERY_WINDOW_S}"
      --push_interval_s="${PUSH_EVAL_INTERVAL_S}"
      --push_magnitudes="${PUSH_EVAL_MAGNITUDES}"
      --output_json="${EVAL_JSON}"
      --output_csv="${EVAL_CSV}"
    )
    if [ "${PUSH_EVAL_DETERMINISTIC}" = "1" ]; then
      EVAL_CMD+=(--deterministic)
    fi
    echo "[pipeline] stage=eval"
    printf '[pipeline] running:'
    printf ' %q' "${EVAL_CMD[@]}"
    printf '\n'
    "${EVAL_CMD[@]}"
  fi
fi

echo "[pipeline] done"
