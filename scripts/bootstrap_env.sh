#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PLAYGROUND_DIR="${ROOT_DIR}/mujoco_playground"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PLAYGROUND_REF="${PLAYGROUND_REF:-f2159f3}"
USE_CUDA="${USE_CUDA:-1}"
BOOTSTRAP_OFFLINE="${BOOTSTRAP_OFFLINE:-0}"

# Some HPC images ship very old git versions without "-C".
if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_SUPPORTS_C=1
else
  GIT_SUPPORTS_C=0
fi

git_in_repo() {
  local repo="$1"
  shift
  if [ "${GIT_SUPPORTS_C}" = "1" ]; then
    git -C "${repo}" "$@"
  else
    (
      cd "${repo}"
      git "$@"
    )
  fi
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN to a valid Python 3.10+ binary."
  exit 1
fi

echo "[bootstrap] root      : ${ROOT_DIR}"
echo "[bootstrap] python    : ${PYTHON_BIN}"
echo "[bootstrap] venv      : ${VENV_DIR}"
echo "[bootstrap] ref       : ${PLAYGROUND_REF}"
echo "[bootstrap] use_cuda  : ${USE_CUDA}"
echo "[bootstrap] offline   : ${BOOTSTRAP_OFFLINE}"

if command -v g++ >/dev/null 2>&1; then
  echo "[bootstrap] g++       : $(command -v g++)"
  echo "[bootstrap] g++ ver   : $(g++ --version | head -n 1)"
fi

if [ "${BOOTSTRAP_OFFLINE}" = "1" ]; then
  if [ ! -d "${VENV_DIR}" ] || [ ! -d "${PLAYGROUND_DIR}" ]; then
    echo "ERROR: offline bootstrap requested but setup is incomplete."
    echo "Run once on login node with BOOTSTRAP_OFFLINE=0:"
    echo "  BOOTSTRAP_OFFLINE=0 bash scripts/bootstrap_env.sh"
    exit 1
  fi
  source "${VENV_DIR}/bin/activate"
  python -c "import mujoco_playground; print('offline bootstrap ok')" >/dev/null
  echo "[bootstrap] offline validation passed"
  exit 0
fi

# Some dependencies (e.g. ml_dtypes fallback build) require C++17 support.
if command -v g++ >/dev/null 2>&1; then
  if ! echo "int main(){return 0;}" | g++ -std=c++17 -x c++ - -o /tmp/cxx17_test_$$ >/dev/null 2>&1; then
    echo "ERROR: g++ does not support -std=c++17 on this login node."
    echo "Load a newer compiler module and retry, for example:"
    echo "  module load compiler/gcc/7.3.0"
    echo "Then re-run:"
    echo "  BOOTSTRAP_OFFLINE=0 bash scripts/bootstrap_env.sh"
    exit 1
  fi
  rm -f /tmp/cxx17_test_$$ || true
fi

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [ ! -d "${PLAYGROUND_DIR}/.git" ]; then
  git clone https://github.com/google-deepmind/mujoco_playground.git "${PLAYGROUND_DIR}"
fi

git_in_repo "${PLAYGROUND_DIR}" fetch --all --tags
git_in_repo "${PLAYGROUND_DIR}" checkout "${PLAYGROUND_REF}"

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
