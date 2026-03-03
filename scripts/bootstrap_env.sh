#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PLAYGROUND_DIR="${ROOT_DIR}/mujoco_playground"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PLAYGROUND_REF="${PLAYGROUND_REF:-f2159f3}"
USE_CUDA="${USE_CUDA:-1}"
BOOTSTRAP_OFFLINE="${BOOTSTRAP_OFFLINE:-0}"
ML_DTYPES_VERSION="${ML_DTYPES_VERSION:-0.5.1}"
REQUIRE_CXX17="${REQUIRE_CXX17:-0}"
PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
PLAYGROUND_INSTALL_MODE="${PLAYGROUND_INSTALL_MODE:-no_warp}"
MUJOCO_VERSION="${MUJOCO_VERSION:-3.3.4}"
MUJOCO_MJX_VERSION="${MUJOCO_MJX_VERSION:-3.3.4}"
INSTALL_WANDB="${INSTALL_WANDB:-0}"

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
echo "[bootstrap] ml_dtypes : ${ML_DTYPES_VERSION}"
echo "[bootstrap] cxx17 req : ${REQUIRE_CXX17}"
echo "[bootstrap] pip cache : ${PIP_NO_CACHE_DIR}"
echo "[bootstrap] mode      : ${PLAYGROUND_INSTALL_MODE}"
echo "[bootstrap] mujoco    : ${MUJOCO_VERSION}"
echo "[bootstrap] mjx       : ${MUJOCO_MJX_VERSION}"
echo "[bootstrap] wandb     : ${INSTALL_WANDB}"

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

# Keep installs quota-friendly on shared HPC filesystems.
export PIP_DISABLE_PIP_VERSION_CHECK=1
PIP_FLAGS=()
if [ "${PIP_NO_CACHE_DIR}" = "1" ]; then
  PIP_FLAGS+=(--no-cache-dir)
fi

# Some dependencies may require C++17 if pip falls back to source builds.
if command -v g++ >/dev/null 2>&1; then
  if ! echo "int main(){return 0;}" | g++ -std=c++17 -x c++ - -o /tmp/cxx17_test_$$ >/dev/null 2>&1; then
    if [ "${REQUIRE_CXX17}" = "1" ]; then
      echo "ERROR: g++ does not support -std=c++17 on this login node."
      echo "Load a newer compiler module and retry."
      exit 1
    else
      echo "[bootstrap] warning: g++ lacks C++17. Using binary-wheel path only."
      echo "[bootstrap] warning: if install later falls back to source builds, load newer GCC and retry."
    fi
  else
    rm -f /tmp/cxx17_test_$$ || true
  fi
fi

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install "${PIP_FLAGS[@]}" --upgrade pip setuptools wheel

if [ ! -d "${PLAYGROUND_DIR}/.git" ]; then
  git clone https://github.com/google-deepmind/mujoco_playground.git "${PLAYGROUND_DIR}"
fi

git_in_repo "${PLAYGROUND_DIR}" fetch --all --tags
git_in_repo "${PLAYGROUND_DIR}" checkout "${PLAYGROUND_REF}"

python -m pip install "${PIP_FLAGS[@]}" --upgrade --prefer-binary --only-binary=ml_dtypes "ml_dtypes==${ML_DTYPES_VERSION}"

if [ "${USE_CUDA}" = "1" ]; then
  python -m pip install "${PIP_FLAGS[@]}" --upgrade --prefer-binary --only-binary=ml_dtypes "jax[cuda12]" "ml_dtypes==${ML_DTYPES_VERSION}"
else
  python -m pip install "${PIP_FLAGS[@]}" --upgrade --prefer-binary --only-binary=ml_dtypes jax "ml_dtypes==${ML_DTYPES_VERSION}"
fi

if [ "${PLAYGROUND_INSTALL_MODE}" = "full" ]; then
  python -m pip install \
    "${PIP_FLAGS[@]}" \
    --prefer-binary \
    --only-binary=ml_dtypes \
    --extra-index-url https://py.mujoco.org \
    --extra-index-url https://pypi.nvidia.com \
    -e "${PLAYGROUND_DIR}[learning]"
else
  # Old enterprise Linux nodes often cannot install warp-lang/mujoco>=3.5 wheels.
  # This path installs a JAX-only stack for train_jax_ppo.py.
  python -m pip install \
    "${PIP_FLAGS[@]}" \
    --prefer-binary \
    --only-binary=ml_dtypes \
    --extra-index-url https://py.mujoco.org \
    "mujoco==${MUJOCO_VERSION}" \
    "mujoco-mjx==${MUJOCO_MJX_VERSION}"

  python -m pip install \
    "${PIP_FLAGS[@]}" \
    --prefer-binary \
    "absl-py" \
    "brax>=0.12.5" \
    "etils" \
    "flax" \
    "lxml" \
    "mediapy" \
    "ml_collections" \
    "orbax-checkpoint>=0.11.22" \
    "tensorboardX" \
    "tqdm"

  if [ "${INSTALL_WANDB}" = "1" ]; then
    python -m pip install "${PIP_FLAGS[@]}" --prefer-binary wandb
  fi

  python -m pip install "${PIP_FLAGS[@]}" --no-deps -e "${PLAYGROUND_DIR}"
fi

echo "[bootstrap] done"
echo "[bootstrap] activate with: source ${VENV_DIR}/bin/activate"
