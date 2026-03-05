#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${ROOT_DIR}/mujoco_playground/logs"

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

  ckpt_dir="${run_dir}/checkpoints"
  [ -d "${ckpt_dir}" ] || continue

  latest_step="$(find "${ckpt_dir}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -n 1)"
  if [ -n "${latest_step}" ]; then
    latest_ckpt="${ckpt_dir}/${latest_step}"
    break
  fi
done < <(find "${LOG_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'G1JoystickRoughTerrain-*' | sort -r)

if [ -z "${latest_ckpt}" ]; then
  echo "ERROR: no valid non-push rough checkpoint found in ${LOG_ROOT}" >&2
  exit 1
fi

echo "${latest_ckpt}"
