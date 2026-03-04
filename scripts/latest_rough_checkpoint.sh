#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${ROOT_DIR}/mujoco_playground/logs"

if [ ! -d "${LOG_ROOT}" ]; then
  echo "ERROR: log directory not found: ${LOG_ROOT}" >&2
  exit 1
fi

latest_run="$(find "${LOG_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'G1JoystickRoughTerrain-*' | sort | tail -n 1)"
if [ -z "${latest_run}" ]; then
  echo "ERROR: no rough-terrain runs found in ${LOG_ROOT}" >&2
  exit 1
fi

ckpt_dir="${latest_run}/checkpoints"
if [ ! -d "${ckpt_dir}" ]; then
  echo "ERROR: checkpoint directory not found: ${ckpt_dir}" >&2
  exit 1
fi

latest_step="$(find "${ckpt_dir}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+$' | sort -n | tail -n 1)"
if [ -z "${latest_step}" ]; then
  echo "ERROR: no step directories found in ${ckpt_dir}" >&2
  exit 1
fi

echo "${ckpt_dir}/${latest_step}"
