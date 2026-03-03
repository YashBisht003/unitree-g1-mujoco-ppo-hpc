#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PLAYGROUND_DIR="${ROOT_DIR}/mujoco_playground"
MENAGERIE_DIR="${PLAYGROUND_DIR}/mujoco_playground/external_deps/mujoco_menagerie"
MENAGERIE_URL="${MENAGERIE_URL:-https://github.com/deepmind/mujoco_menagerie.git}"
MENAGERIE_COMMIT="${MENAGERIE_COMMIT:-1b86ece576591213e2b666ebf59508454200ca97}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
# Pinned to pre-contact-sensor commit for MuJoCo 3.3.x compatibility.
PLAYGROUND_REF="${PLAYGROUND_REF:-d886c80}"
USE_CUDA="${USE_CUDA:-1}"
BOOTSTRAP_OFFLINE="${BOOTSTRAP_OFFLINE:-0}"
ML_DTYPES_VERSION="${ML_DTYPES_VERSION:-0.5.1}"
JAX_VERSION="${JAX_VERSION:-0.5.3}"
JAXLIB_VERSION="${JAXLIB_VERSION:-${JAX_VERSION}}"
JAX_CUDA_EXTRA="${JAX_CUDA_EXTRA:-cuda12}"
JAX_CUDA11_WHEELS_URL="${JAX_CUDA11_WHEELS_URL:-https://storage.googleapis.com/jax-releases/jax_cuda_releases.html}"
BRAX_VERSION="${BRAX_VERSION:-0.12.5}"
FLAX_VERSION="${FLAX_VERSION:-0.10.6}"
ORBAX_VERSION="${ORBAX_VERSION:-0.11.22}"
REQUIRE_CXX17="${REQUIRE_CXX17:-0}"
PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
PLAYGROUND_INSTALL_MODE="${PLAYGROUND_INSTALL_MODE:-no_warp}"
MUJOCO_VERSION="${MUJOCO_VERSION:-3.3.4}"
MUJOCO_MJX_VERSION="${MUJOCO_MJX_VERSION:-3.3.4}"
INSTALL_WANDB="${INSTALL_WANDB:-0}"
USE_MEDIAPY_SHIM="${USE_MEDIAPY_SHIM:-1}"
USE_WANDB_SHIM="${USE_WANDB_SHIM:-1}"

# Some HPC images ship very old git versions without "-C".
if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_SUPPORTS_C=1
else
  GIT_SUPPORTS_C=0
fi
GIT_USER_NAME="${GIT_USER_NAME:-hpc-user}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-hpc-user@localhost}"
GIT_DISABLE_REFLOG="${GIT_DISABLE_REFLOG:-1}"
GIT_ARGS=(-c "user.name=${GIT_USER_NAME}" -c "user.email=${GIT_USER_EMAIL}")
if [ "${GIT_DISABLE_REFLOG}" = "1" ]; then
  GIT_ARGS+=(-c "core.logAllRefUpdates=false")
fi

git_in_repo() {
  local repo="$1"
  shift
  if [ "${GIT_SUPPORTS_C}" = "1" ]; then
    git "${GIT_ARGS[@]}" -C "${repo}" "$@"
  else
    (
      cd "${repo}"
      git "${GIT_ARGS[@]}" "$@"
    )
  fi
}

apply_mjx_make_data_compat_shim() {
  local target="${PLAYGROUND_DIR}/mujoco_playground/_src/mjx_env.py"
  if [ ! -f "${target}" ]; then
    echo "[bootstrap] warning: ${target} not found; skipping mjx make_data compat shim."
    return 0
  fi

  "${PYTHON_BIN}" - "${target}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
marker = "__codex_mjx_make_data_compat__"
if marker in text:
    print("[bootstrap] mjx make_data compat shim already present")
    raise SystemExit(0)

pattern = r'(?m)^(?P<indent>\s*)data = mjx\.make_data\(model, impl=impl, nconmax=nconmax, njmax=njmax\)\s*$'
match = re.search(pattern, text)
if not match:
    print("[bootstrap] mjx make_data compat target not found; leaving file unchanged")
    raise SystemExit(0)

indent = match.group("indent")
replacement = (
    f"{indent}# {marker}\n"
    f"{indent}try:\n"
    f"{indent}  data = mjx.make_data(model, impl=impl, nconmax=nconmax, njmax=njmax)\n"
    f"{indent}except TypeError:\n"
    f"{indent}  try:\n"
    f"{indent}    data = mjx.make_data(model, impl=impl)\n"
    f"{indent}  except TypeError:\n"
    f"{indent}    data = mjx.make_data(model)\n"
)
text = re.sub(pattern, replacement, text, count=1)
path.write_text(text)
print(f"[bootstrap] installed mjx make_data compat shim at {path}")
PY
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN to a valid Python 3.10+ binary."
  exit 1
fi

# Prevent inherited conda/python env vars from polluting this venv.
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE __PYVENV_LAUNCHER__ || true
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL _CE_M _CE_CONDA || true

PY_VER="$("${PYTHON_BIN}" - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:3])))
PY
)"
PY_OK="$("${PYTHON_BIN}" - <<'PY'
import sys
print(1 if sys.version_info >= (3, 10) else 0)
PY
)"
if [ "${PY_OK}" != "1" ]; then
  echo "ERROR: ${PYTHON_BIN} is Python ${PY_VER}, but Python >= 3.10 is required."
  echo "Use your conda/env python explicitly, e.g.:"
  echo "  conda activate vlm_new"
  echo "  PYTHON_BIN=$(which python) BOOTSTRAP_OFFLINE=0 bash scripts/bootstrap_env.sh"
  exit 1
fi

echo "[bootstrap] root      : ${ROOT_DIR}"
echo "[bootstrap] python    : ${PYTHON_BIN}"
echo "[bootstrap] venv      : ${VENV_DIR}"
echo "[bootstrap] ref       : ${PLAYGROUND_REF}"
echo "[bootstrap] menagerie : ${MENAGERIE_DIR}"
echo "[bootstrap] menag url : ${MENAGERIE_URL}"
echo "[bootstrap] menag sha : ${MENAGERIE_COMMIT}"
echo "[bootstrap] use_cuda  : ${USE_CUDA}"
echo "[bootstrap] offline   : ${BOOTSTRAP_OFFLINE}"
echo "[bootstrap] ml_dtypes : ${ML_DTYPES_VERSION}"
echo "[bootstrap] jax       : ${JAX_VERSION}"
echo "[bootstrap] jaxlib    : ${JAXLIB_VERSION}"
echo "[bootstrap] jax extra : ${JAX_CUDA_EXTRA}"
echo "[bootstrap] cuda11 whl: ${JAX_CUDA11_WHEELS_URL}"
echo "[bootstrap] brax      : ${BRAX_VERSION}"
echo "[bootstrap] flax      : ${FLAX_VERSION}"
echo "[bootstrap] orbax     : ${ORBAX_VERSION}"
echo "[bootstrap] cxx17 req : ${REQUIRE_CXX17}"
echo "[bootstrap] pip cache : ${PIP_NO_CACHE_DIR}"
echo "[bootstrap] mode      : ${PLAYGROUND_INSTALL_MODE}"
echo "[bootstrap] mujoco    : ${MUJOCO_VERSION}"
echo "[bootstrap] mjx       : ${MUJOCO_MJX_VERSION}"
echo "[bootstrap] wandb     : ${INSTALL_WANDB}"
echo "[bootstrap] media shim: ${USE_MEDIAPY_SHIM}"
echo "[bootstrap] wandb shim: ${USE_WANDB_SHIM}"
echo "[bootstrap] git ident : ${GIT_USER_NAME} <${GIT_USER_EMAIL}> reflog_off=${GIT_DISABLE_REFLOG}"

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
  if [ ! -d "${MENAGERIE_DIR}" ]; then
    echo "ERROR: offline bootstrap requested but menagerie is missing:"
    echo "  ${MENAGERIE_DIR}"
    echo "Run once on login node with BOOTSTRAP_OFFLINE=0 to pre-download assets."
    exit 1
  fi
  apply_mjx_make_data_compat_shim
  source "${VENV_DIR}/bin/activate"
  if [ "${USE_CUDA}" != "1" ]; then
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
    export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
  fi
  python -c "import mujoco_playground; print('offline bootstrap ok')" >/dev/null
  echo "[bootstrap] offline validation passed"
  exit 0
fi

# Keep installs quota-friendly on shared HPC filesystems.
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1
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
export PYTHONPYCACHEPREFIX="${VENV_DIR}/.pycache"

# Prevent incompatible core pin sets from reaching a confusing pip failure.
"${PYTHON_BIN}" - "${JAX_VERSION}" "${FLAX_VERSION}" <<'PY'
import sys

def parse(v: str):
  parts = []
  for token in v.split("."):
    digits = "".join(ch for ch in token if ch.isdigit())
    parts.append(int(digits) if digits else 0)
  while len(parts) < 3:
    parts.append(0)
  return tuple(parts[:3])

jax_v = parse(sys.argv[1])
flax_v = parse(sys.argv[2])

if jax_v < (0, 5, 1) and flax_v >= (0, 10, 6):
  print("ERROR: incompatible pins: flax>=0.10.6 requires jax>=0.5.1.")
  print("Current values: JAX_VERSION=%s FLAX_VERSION=%s" % (sys.argv[1], sys.argv[2]))
  print("Do one of these:")
  print("  1) Keep JAX_VERSION>=0.5.1 (current default path)")
  print("  2) Move to a full legacy stack branch (older flax/brax/playground pins)")
  raise SystemExit(1)
PY

python -m pip install "${PIP_FLAGS[@]}" --upgrade pip setuptools wheel

if [ ! -d "${PLAYGROUND_DIR}/.git" ]; then
  git "${GIT_ARGS[@]}" clone https://github.com/google-deepmind/mujoco_playground.git "${PLAYGROUND_DIR}"
fi

git_in_repo "${PLAYGROUND_DIR}" fetch --all --tags
git_in_repo "${PLAYGROUND_DIR}" checkout "${PLAYGROUND_REF}"
apply_mjx_make_data_compat_shim

python -m pip install "${PIP_FLAGS[@]}" --upgrade --prefer-binary --only-binary=ml_dtypes "ml_dtypes==${ML_DTYPES_VERSION}"

if [ "${USE_CUDA}" = "1" ]; then
  if [ "${JAX_CUDA_EXTRA}" = "cuda11_pip" ]; then
    python -m pip install "${PIP_FLAGS[@]}" --upgrade --prefer-binary --only-binary=ml_dtypes \
      --find-links "${JAX_CUDA11_WHEELS_URL}" \
      "jax[${JAX_CUDA_EXTRA}]==${JAX_VERSION}" "jax==${JAX_VERSION}" "ml_dtypes==${ML_DTYPES_VERSION}"
  else
    python -m pip install "${PIP_FLAGS[@]}" --upgrade --prefer-binary --only-binary=ml_dtypes \
      "jax[${JAX_CUDA_EXTRA}]==${JAX_VERSION}" "jax==${JAX_VERSION}" "jaxlib==${JAXLIB_VERSION}" "ml_dtypes==${ML_DTYPES_VERSION}"
  fi
else
  python -m pip install "${PIP_FLAGS[@]}" --upgrade --prefer-binary --only-binary=ml_dtypes \
    "jax==${JAX_VERSION}" "jaxlib==${JAXLIB_VERSION}" "ml_dtypes==${ML_DTYPES_VERSION}"
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
    "brax==${BRAX_VERSION}" \
    "etils" \
    "flax==${FLAX_VERSION}" \
    "lxml" \
    "mediapy" \
    "ml_collections" \
    "orbax-checkpoint==${ORBAX_VERSION}" \
    "tensorboardX" \
    "tqdm"

  if [ "${INSTALL_WANDB}" = "1" ]; then
    python -m pip install "${PIP_FLAGS[@]}" --prefer-binary wandb
  fi

  python -m pip install "${PIP_FLAGS[@]}" --no-deps -e "${PLAYGROUND_DIR}"
fi

if [ "${USE_MEDIAPY_SHIM}" = "1" ]; then
  cat > "${PLAYGROUND_DIR}/learning/mediapy.py" <<'PY'
"""Headless mediapy shim for cluster training."""

def write_video(*args, **kwargs):
  # No-op on HPC; avoids importing IPython in real mediapy.
  print("[mediapy-shim] write_video skipped.")
PY
  echo "[bootstrap] installed mediapy shim at ${PLAYGROUND_DIR}/learning/mediapy.py"
fi

if [ "${USE_WANDB_SHIM}" = "1" ] && [ "${INSTALL_WANDB}" != "1" ]; then
  cat > "${PLAYGROUND_DIR}/learning/wandb.py" <<'PY'
"""Headless wandb shim for cluster training when --use_wandb=False."""


class _Run:
  def log(self, *args, **kwargs):
    return None

  def finish(self, *args, **kwargs):
    return None

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    return False


run = _Run()
config = {}
summary = {}


def init(*args, **kwargs):
  return run


def log(*args, **kwargs):
  return None


def finish(*args, **kwargs):
  return None
PY
  echo "[bootstrap] installed wandb shim at ${PLAYGROUND_DIR}/learning/wandb.py"
fi

# Pre-download menagerie on login node so compute-node jobs can run fully offline.
if [ ! -d "${MENAGERIE_DIR}/.git" ]; then
  mkdir -p "$(dirname "${MENAGERIE_DIR}")"
  git "${GIT_ARGS[@]}" clone --depth 1 "${MENAGERIE_URL}" "${MENAGERIE_DIR}"
fi
git_in_repo "${MENAGERIE_DIR}" fetch --all --tags || true
git_in_repo "${MENAGERIE_DIR}" checkout "${MENAGERIE_COMMIT}"
echo "[bootstrap] menagerie ready at ${MENAGERIE_DIR}"

echo "[bootstrap] done"
echo "[bootstrap] activate with: source ${VENV_DIR}/bin/activate"
