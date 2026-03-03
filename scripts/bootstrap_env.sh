#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PLAYGROUND_DIR="${ROOT_DIR}/mujoco_playground"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PLAYGROUND_REF="${PLAYGROUND_REF:-f2159f3}"
USE_CUDA="${USE_CUDA:-1}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN to a valid Python 3.10+ binary."
  exit 1
fi

echo "[bootstrap] root      : ${ROOT_DIR}"
echo "[bootstrap] python    : ${PYTHON_BIN}"
echo "[bootstrap] venv      : ${VENV_DIR}"
echo "[bootstrap] ref       : ${PLAYGROUND_REF}"
echo "[bootstrap] use_cuda  : ${USE_CUDA}"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [ ! -d "${PLAYGROUND_DIR}/.git" ]; then
  git clone https://github.com/google-deepmind/mujoco_playground.git "${PLAYGROUND_DIR}"
fi

git -C "${PLAYGROUND_DIR}" fetch --all --tags
git -C "${PLAYGROUND_DIR}" checkout "${PLAYGROUND_REF}"

if [ "${USE_CUDA}" = "1" ]; then
  python -m pip install --upgrade "jax[cuda12]"
else
  python -m pip install --upgrade jax
fi

python -m pip install \
  --extra-index-url https://py.mujoco.org \
  --extra-index-url https://pypi.nvidia.com \
  -e "${PLAYGROUND_DIR}[learning]"

echo "[bootstrap] done"
echo "[bootstrap] activate with: source ${VENV_DIR}/bin/activate"
