#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source .venv/bin/activate
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${ROOT_DIR}/logs/angle_sweep_body_singlepush_baseline_${TS}.log"
JSON="${ROOT_DIR}/research/results/angle_sweep_body_singlepush_baseline_${TS}.json"
CSV="${ROOT_DIR}/research/results/angle_sweep_body_singlepush_baseline_${TS}.csv"

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/research/results"

echo "[baseline-sweep] log=${LOG}"
echo "[baseline-sweep] json=${JSON}"
echo "[baseline-sweep] csv=${CSV}"

python -u research/eval_recovery_rate.py \
  --env_name G1JoystickFlatTerrain \
  --checkpoint_path "${ROOT_DIR}/mujoco_playground/logs/G1JoystickRoughTerrain-20260322-155050-g1-push-single_gpu-baseline-from-152535040/checkpoints/000028016640" \
  --command 0.0,0.0,0.0 \
  --push_magnitudes 1,1.5,2,2.25,2.5,2.75,3 \
  --push_angles_deg 0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345 \
  --push_angle_frame body \
  --single_push \
  --push_interval_s 2.0 \
  --episodes_per_magnitude 20 \
  --batch_size 4 \
  --episode_length 400 \
  --recovery_window_s 5.0 \
  --deterministic \
  --playground_config_overrides '{"noise_config":{"level":0.0,"scales":{"joint_pos":0.03,"joint_vel":1.5,"gravity":0.05,"linvel":0.1,"gyro":0.2}}}' \
  --output_json "${JSON}" \
  --output_csv "${CSV}" \
  2>&1 | tee "${LOG}"
