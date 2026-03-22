#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${ROOT_DIR}/.venv/bin/python"

if [ ! -x "${VENV_PY}" ]; then
  echo "ERROR: missing ${VENV_PY}. Run: bash scripts/bootstrap_env.sh" >&2
  exit 1
fi

INPUT_PATH="${1:-${CHECKPOINT_PATH:-}}"
if [ -z "${INPUT_PATH}" ]; then
  echo "ERROR: checkpoint path argument is required." >&2
  exit 1
fi

ABS_INPUT="$("${VENV_PY}" - <<'PY' "${INPUT_PATH}"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

if [ -d "${ABS_INPUT}/checkpoints" ]; then
  SEARCH_ROOT="${ABS_INPUT}/checkpoints"
  REQUESTED_STEP=""
elif [ -d "${ABS_INPUT}" ] && echo "$(basename "${ABS_INPUT}")" | grep -Eq '^[0-9]+$'; then
  SEARCH_ROOT="$(dirname "${ABS_INPUT}")"
  REQUESTED_STEP="$(basename "${ABS_INPUT}")"
elif [ -d "${ABS_INPUT}" ]; then
  SEARCH_ROOT="${ABS_INPUT}"
  REQUESTED_STEP=""
else
  echo "ERROR: checkpoint path does not exist: ${ABS_INPUT}" >&2
  exit 1
fi

if [ ! -d "${SEARCH_ROOT}" ]; then
  echo "ERROR: checkpoint search root does not exist: ${SEARCH_ROOT}" >&2
  exit 1
fi

mapfile -t STEP_NAMES < <(
  find "${SEARCH_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null \
    | grep -E '^[0-9]+$' \
    | sort -nr
)

if [ "${#STEP_NAMES[@]}" -eq 0 ]; then
  echo "ERROR: no numeric checkpoint step directories found in ${SEARCH_ROOT}" >&2
  exit 1
fi

for STEP_NAME in "${STEP_NAMES[@]}"; do
  if [ -n "${REQUESTED_STEP}" ] && [ $((10#${STEP_NAME})) -gt $((10#${REQUESTED_STEP})) ]; then
    continue
  fi
  STEP_DIR="${SEARCH_ROOT}/${STEP_NAME}"
  if "${VENV_PY}" - <<'PY' "${STEP_DIR}" >/dev/null 2>&1
from pathlib import Path
import sys
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

ckpt = Path(sys.argv[1])
ppo_checkpoint.load_config(ckpt)
ppo_checkpoint.load(ckpt)
PY
  then
    printf '%s\n' "${STEP_DIR}"
    exit 0
  fi
  echo "[resolve-ckpt] skipped unreadable checkpoint: ${STEP_DIR}" >&2
done

echo "ERROR: no restorable checkpoint found under ${SEARCH_ROOT}" >&2
exit 1
