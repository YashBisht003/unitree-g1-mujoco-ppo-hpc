#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${ROOT_DIR}/mujoco_playground/logs"
RESOLVE_CKPT_SCRIPT="${ROOT_DIR}/scripts/resolve_ppo_checkpoint.sh"

if [ ! -d "${LOG_ROOT}" ]; then
  echo "ERROR: log directory not found: ${LOG_ROOT}" >&2
  exit 1
fi

while IFS= read -r run_dir; do
  if ckpt="$("${RESOLVE_CKPT_SCRIPT}" "${run_dir}" 2>/dev/null)"; then
    echo "${ckpt}"
    exit 0
  fi
done < <(find "${LOG_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'G1JoystickFlatTerrain-*' | sort -r)

echo "ERROR: no valid flat-terrain checkpoint found in ${LOG_ROOT}" >&2
exit 1
