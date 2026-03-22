#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${ROOT_DIR}/mujoco_playground/logs"
RESOLVE_CKPT_SCRIPT="${ROOT_DIR}/scripts/resolve_ppo_checkpoint.sh"

if [ ! -d "${LOG_ROOT}" ]; then
  echo "ERROR: log directory not found: ${LOG_ROOT}" >&2
  exit 1
fi

latest_ckpt=""
while IFS= read -r run_dir; do
  # Exclude push-recovery runs; we want a base rough checkpoint source.
  if echo "${run_dir}" | grep -q -- "-g1-push-"; then
    continue
  fi

  if latest_ckpt="$("${RESOLVE_CKPT_SCRIPT}" "${run_dir}" 2>/dev/null)"; then
    break
  fi
done < <(find "${LOG_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'G1JoystickRoughTerrain-*' | sort -r)

if [ -z "${latest_ckpt}" ]; then
  echo "ERROR: no valid non-push rough checkpoint found in ${LOG_ROOT}" >&2
  exit 1
fi

echo "${latest_ckpt}"
