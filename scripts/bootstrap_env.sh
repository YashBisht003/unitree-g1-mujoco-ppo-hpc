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
NUMPY_VERSION="${NUMPY_VERSION:-}"
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

apply_jax_clip_kwarg_compat_shim() {
  local src_dir="${PLAYGROUND_DIR}/mujoco_playground/_src"
  if [ ! -d "${src_dir}" ]; then
    echo "[bootstrap] warning: ${src_dir} not found; skipping clip kwarg compat shim."
    return 0
  fi

  "${PYTHON_BIN}" - "${src_dir}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])


def rewrite_clip_calls(text: str):
    tokens = ("jp.clip(", "jnp.clip(")
    out = []
    i = 0
    total_changes = 0
    n = len(text)

    while i < n:
        next_hits = []
        for tok in tokens:
            pos = text.find(tok, i)
            if pos != -1:
                next_hits.append((pos, tok))
        if not next_hits:
            out.append(text[i:])
            break

        pos, tok = min(next_hits, key=lambda item: item[0])
        out.append(text[i : pos + len(tok)])

        args_start = pos + len(tok)
        depth = 1
        j = args_start
        while j < n and depth > 0:
            ch = text[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1

        if depth != 0:
            out.append(text[args_start:])
            i = n
            break

        args = text[args_start : j - 1]
        new_args = re.sub(r"(?<![A-Za-z0-9_])min\s*=", "a_min=", args)
        new_args = re.sub(r"(?<![A-Za-z0-9_])max\s*=", "a_max=", new_args)
        if new_args != args:
            total_changes += 1
        out.append(new_args)
        out.append(")")
        i = j

    return "".join(out), total_changes


changed_files = 0
changed_calls = 0
for path in root.rglob("*.py"):
    text = path.read_text()
    new_text, delta = rewrite_clip_calls(text)
    if delta:
        path.write_text(new_text)
        changed_files += 1
        changed_calls += delta

if changed_files:
    print(
        f"[bootstrap] installed clip kwarg compat shim in {changed_files} file(s), {changed_calls} call(s)"
    )
else:
    print("[bootstrap] clip kwarg compat shim already satisfied")
PY
}

apply_maxrl_scaffold_shim() {
  local target="${PLAYGROUND_DIR}/learning/train_jax_ppo.py"
  if [ ! -f "${target}" ]; then
    echo "[bootstrap] warning: ${target} not found; skipping MaxRL scaffold shim."
    return 0
  fi

  "${PYTHON_BIN}" - "${target}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
marker = "__codex_maxrl_scaffold_v6__"
if marker in text:
  print("[bootstrap] MaxRL scaffold shim already present")
  raise SystemExit(0)

if "from brax.training.agents.ppo import losses as ppo_losses" not in text:
  text = text.replace(
      "from brax.training.agents.ppo import train as ppo\n",
      "from brax.training.agents.ppo import losses as ppo_losses\n"
      "from brax.training.agents.ppo import train as ppo\n",
      1,
  )

flags_block = """
# __MARKER__
_ADV_MODE = flags.DEFINE_enum(
    "adv_mode",
    "ppo",
    ["ppo", "maxrl_binary", "maxrl_temporal"],
    "Advantage mode scaffold: baseline PPO, MaxRL-binary, or temporal-binary.",
)
_SCENARIO_GROUP_SIZE = flags.DEFINE_integer(
    "scenario_group_size",
    0,
    "If >0, enforce fixed rollout grouping (num_envs must be divisible).",
)
_MAXRL_LOG_ONLY = flags.DEFINE_boolean(
    "maxrl_log_only",
    False,
    "If true, emit MaxRL diagnostics only and keep baseline PPO loss.",
)
_MAXRL_SCENARIO_KEY = flags.DEFINE_string(
    "maxrl_scenario_key",
    "",
    "Optional scenario descriptor printed in MaxRL scaffold logs.",
)
_MAXRL_VERBOSE = flags.DEFINE_boolean(
    "maxrl_verbose",
    False,
    "Enable verbose MaxRL scaffold logging.",
)
_MAXRL_EPISODE_VERIFIER = flags.DEFINE_boolean(
    "maxrl_episode_verifier",
    True,
    "Use episode_done-aware verifier for maxrl_binary (Option A).",
)
"""
flags_block = flags_block.replace("__MARKER__", marker)

helpers_block = """
# __MARKER__
def _groupwise_binary_weights(success, group_size: int, valid=None):
  \"\"\"Computes per-rollout MaxRL weights inside scenario groups.

  Args:
    success: shape [batch], in [0, 1], success indicator/rate.
    group_size: scenario rollout group size.
    valid: optional shape [batch], 1 where success is episode-level valid.
      Invalid entries fall back to PPO weight=1.
  \"\"\"

  batch = int(success.shape[0])
  if valid is None:
    valid = jp.ones_like(success)
  else:
    valid = valid.astype(success.dtype)

  success = success * valid
  if group_size <= 0 or batch % group_size != 0:
    k = jp.sum(success)
    v = jp.sum(valid)
    maxrl_w = jp.where(
        k > 0,
        success / (k + 1e-8),
        jp.where(v > 0, valid / (v + 1e-8), jp.zeros_like(success)),
    )
    return jp.where(valid > 0, maxrl_w, jp.ones_like(success))

  grouped_s = jp.reshape(success, (-1, group_size))
  grouped_v = jp.reshape(valid, (-1, group_size))
  k = jp.sum(grouped_s, axis=1, keepdims=True)
  v = jp.sum(grouped_v, axis=1, keepdims=True)
  grouped_maxrl_w = jp.where(
      k > 0,
      grouped_s / (k + 1e-8),
      jp.where(v > 0, grouped_v / (v + 1e-8), jp.zeros_like(grouped_s)),
  )
  grouped_w = jp.where(grouped_v > 0, grouped_maxrl_w, jp.ones_like(grouped_s))
  return jp.reshape(grouped_w, (batch,))


def _episode_success_from_episode_done(termination, episode_done):
  \"\"\"Computes episode-level success from done events inside a chunk.

  This makes MaxRL-binary align with full-episode outcome signals (Option A)
  instead of only unroll-window termination.
  \"\"\"

  episode_done = episode_done.astype(termination.dtype)

  def _scan_fn(carry, inputs):
    ever_terminated = carry
    term_t, done_t = inputs
    ever_terminated = jp.maximum(ever_terminated, term_t)
    success_t = (1.0 - ever_terminated) * done_t
    # Start a fresh episode immediately after done.
    ever_terminated = jp.where(done_t > 0, jp.zeros_like(ever_terminated), ever_terminated)
    return ever_terminated, (done_t, success_t)

  init_ever_terminated = jp.zeros_like(termination[0])
  _, (done_hist, success_hist) = jax.lax.scan(
      _scan_fn, init_ever_terminated, (termination, episode_done)
  )
  done_count = jp.sum(done_hist, axis=0)
  success = jp.where(
      done_count > 0,
      jp.sum(success_hist, axis=0) / (done_count + 1e-8),
      jp.zeros_like(done_count),
  )
  valid = (done_count > 0).astype(termination.dtype)
  return success, valid


def _groupwise_temporal_weights(temporal_success, group_size: int):
  \"\"\"Computes per-(t, rollout) MaxRL-T weights within scenario groups.\"\"\"

  steps = int(temporal_success.shape[0])
  batch = int(temporal_success.shape[1])
  if group_size <= 0 or batch % group_size != 0:
    k = jp.sum(temporal_success, axis=1, keepdims=True)
    # Per-timestep graceful fallback when no rollout is successful.
    return jp.where(
        k > 0,
        temporal_success / (k + 1e-8),
        jp.ones_like(temporal_success) / float(batch),
    )

  # Assumes contiguous scenario grouping in env batching:
  # [0..group_size-1] = scenario 0, [group_size..2*group_size-1] = scenario 1, ...
  grouped = jp.reshape(temporal_success, (steps, -1, group_size))
  k = jp.sum(grouped, axis=2, keepdims=True)
  grouped_w = jp.where(
      k > 0, grouped / (k + 1e-8), jp.ones_like(grouped) / float(group_size)
  )
  return jp.reshape(grouped_w, (steps, batch))


def _compute_ppo_loss_with_maxrl(
    params,
    normalizer_params,
    data,
    rng,
    ppo_network,
    entropy_cost=1e-4,
    discounting=0.9,
    reward_scaling=1.0,
    gae_lambda=0.95,
    clipping_epsilon=0.3,
    normalize_advantage=True,
):
  \"\"\"Drop-in replacement for Brax PPO loss with MaxRL/MaxRL-T reweighting.\"\"\"

  parametric_action_distribution = ppo_network.parametric_action_distribution
  policy_apply = ppo_network.policy_network.apply
  value_apply = ppo_network.value_network.apply

  data = jax.tree_util.tree_map(lambda x: jp.swapaxes(x, 0, 1), data)
  policy_logits = policy_apply(normalizer_params, params.policy, data.observation)

  baseline = value_apply(normalizer_params, params.value, data.observation)
  terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
  bootstrap_value = value_apply(normalizer_params, params.value, terminal_obs)

  rewards = data.reward * reward_scaling
  truncation = data.extras["state_extras"]["truncation"]
  termination = (1 - data.discount) * (1 - truncation)

  target_action_log_probs = parametric_action_distribution.log_prob(
      policy_logits, data.extras["policy_extras"]["raw_action"]
  )
  behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

  vs, advantages = ppo_losses.compute_gae(
      truncation=truncation,
      termination=termination,
      rewards=rewards,
      values=baseline,
      bootstrap_value=bootstrap_value,
      lambda_=gae_lambda,
      discount=discounting,
  )

  maxrl_success_rate = jp.array(0.0, dtype=rewards.dtype)
  maxrl_valid_fraction = jp.array(1.0, dtype=rewards.dtype)
  adv_mode = _ADV_MODE.value
  group_size = int(_SCENARIO_GROUP_SIZE.value)

  if adv_mode == "maxrl_binary" and not _MAXRL_LOG_ONLY.value:
    state_extras = data.extras["state_extras"]
    if _MAXRL_EPISODE_VERIFIER.value and "episode_done" in state_extras:
      rollout_success, rollout_valid = _episode_success_from_episode_done(
          termination, state_extras["episode_done"]
      )
      rollout_weights = _groupwise_binary_weights(
          rollout_success, group_size, rollout_valid
      )
      maxrl_valid_fraction = jp.mean(rollout_valid)
    else:
      # Fallback verifier when episode_done is unavailable.
      rollout_success = 1.0 - jp.max(termination, axis=0)
      rollout_weights = _groupwise_binary_weights(rollout_success, group_size)
    advantages = advantages * rollout_weights[None, :]
    maxrl_success_rate = jp.mean(rollout_success)
  elif adv_mode == "maxrl_temporal" and not _MAXRL_LOG_ONLY.value:
    # Temporal verifier: alive mask per (t, rollout).
    temporal_success = 1.0 - termination
    temporal_weights = _groupwise_temporal_weights(temporal_success, group_size)
    advantages = advantages * temporal_weights
    maxrl_success_rate = jp.mean(temporal_success)
  elif normalize_advantage:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  if normalize_advantage and adv_mode != "ppo" and not _MAXRL_LOG_ONLY.value:
    # Secondary stabilization after MaxRL weighting.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  rho_s = jp.exp(target_action_log_probs - behaviour_action_log_probs)
  surrogate_loss1 = rho_s * advantages
  surrogate_loss2 = jp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
  policy_loss = -jp.mean(jp.minimum(surrogate_loss1, surrogate_loss2))

  v_error = vs - baseline
  v_loss = jp.mean(v_error * v_error) * 0.5 * 0.5

  entropy = jp.mean(parametric_action_distribution.entropy(policy_logits, rng))
  entropy_loss = entropy_cost * -entropy

  total_loss = policy_loss + v_loss + entropy_loss
  metrics = {
      "total_loss": total_loss,
      "policy_loss": policy_loss,
      "v_loss": v_loss,
      "entropy_loss": entropy_loss,
  }
  if adv_mode != "ppo":
    metrics["maxrl/success_rate"] = maxrl_success_rate
    if adv_mode == "maxrl_binary":
      metrics["maxrl/valid_fraction"] = maxrl_valid_fraction
  return total_loss, metrics


def _install_maxrl_loss_patch() -> None:
  \"\"\"Installs runtime PPO-loss patch when adv_mode requests MaxRL behavior.\"\"\"

  adv_mode = _ADV_MODE.value
  if adv_mode == "ppo":
    return
  if _MAXRL_LOG_ONLY.value:
    print("[maxrl] maxrl_log_only=True -> keeping baseline PPO loss.")
    return

  patched_name = getattr(ppo_losses.compute_ppo_loss, "__name__", "")
  if patched_name == "_compute_ppo_loss_with_maxrl":
    return

  ppo_losses.compute_ppo_loss = _compute_ppo_loss_with_maxrl
  print(
      "[maxrl] installed PPO loss patch "
      f"(adv_mode={adv_mode}, scenario_group_size={_SCENARIO_GROUP_SIZE.value})."
  )


def _log_maxrl_scaffold_config(num_envs: int) -> None:
  \"\"\"Validates grouping inputs and prints MaxRL diagnostics.\"\"\"

  adv_mode = _ADV_MODE.value
  group = _SCENARIO_GROUP_SIZE.value
  if group < 0:
    raise ValueError(
        f"Invalid --scenario_group_size={group}. Expected >= 0."
    )

  scenario_count = 0
  if group > 0:
    if num_envs % group != 0:
      raise ValueError(
          "scenario_group_size must divide num_envs exactly: "
          f"num_envs={num_envs}, scenario_group_size={group}"
      )
    scenario_count = num_envs // group
    print(
        "[maxrl] scenario grouping enabled: "
        f"{scenario_count} scenarios x {group} rollouts (num_envs={num_envs})."
    )
  elif adv_mode != "ppo":
    print(
        "[maxrl] warning: adv_mode requested without scenario grouping "
        f"(adv_mode={adv_mode}, scenario_group_size={group})."
    )

  if _MAXRL_SCENARIO_KEY.value:
    print(f"[maxrl] scenario key: {_MAXRL_SCENARIO_KEY.value}")

  if adv_mode != "ppo":
    print(
        "[maxrl] adv_mode active: "
        f"{adv_mode} (set --maxrl_log_only=False to enable objective patch)."
    )

  if _MAXRL_LOG_ONLY.value:
    print("[maxrl] maxrl_log_only=True (diagnostics only).")

  if _MAXRL_VERBOSE.value:
    print(
        "[maxrl] verbose: "
        f"num_envs={num_envs}, scenario_group_size={group}, "
        f"scenario_count={scenario_count}, adv_mode={adv_mode}"
    )
"""
helpers_block = helpers_block.replace("__MARKER__", marker)

if "def get_rl_config(" not in text or "def main(argv):" not in text:
  print("[bootstrap] MaxRL scaffold shim target layout not recognized; skipping")
  raise SystemExit(0)

if "_ADV_MODE = flags.DEFINE_enum(" not in text:
  text, n = re.subn(
      r"\n\ndef get_rl_config\(",
      "\n\n" + flags_block + "\n\ndef get_rl_config(",
      text,
      count=1,
  )
  if n == 0:
    print("[bootstrap] MaxRL scaffold flags insertion failed; skipping")
    raise SystemExit(0)

if "_MAXRL_EPISODE_VERIFIER = flags.DEFINE_boolean(" not in text:
  episode_flag_block = """
_MAXRL_EPISODE_VERIFIER = flags.DEFINE_boolean(
    "maxrl_episode_verifier",
    True,
    "Use episode_done-aware verifier for maxrl_binary (Option A).",
)
"""
  text, n = re.subn(
      r"\n\ndef get_rl_config\(",
      "\n" + episode_flag_block + "\n\ndef get_rl_config(",
      text,
      count=1,
  )
  if n == 0:
    print("[bootstrap] MaxRL episode verifier flag insertion failed; skipping")
    raise SystemExit(0)

text, replaced_helpers = re.subn(
    r"\n# __codex_maxrl_scaffold_v[0-9A-Za-z_]+__\n(?:def _groupwise_binary_weights|def _log_maxrl_scaffold_config)[\s\S]*?\n\ndef main\(argv\):",
    "\n\n" + helpers_block + "\n\ndef main(argv):",
    text,
    count=1,
)

if replaced_helpers == 0:
  text, replaced_helpers = re.subn(
      r"\n\ndef _groupwise_binary_weights\([\s\S]*?\n\ndef main\(argv\):",
      "\n\n" + helpers_block + "\n\ndef main(argv):",
      text,
      count=1,
  )

if replaced_helpers == 0 and "def _groupwise_binary_weights(" not in text and "def _log_maxrl_scaffold_config(" not in text:
  text, n = re.subn(
      r"\n\ndef main\(argv\):",
      "\n\n" + helpers_block + "\n\ndef main(argv):",
      text,
      count=1,
  )
  if n == 0:
    print("[bootstrap] MaxRL scaffold helper insertion failed; skipping")
    raise SystemExit(0)

call_anchor = '  print(f"PPO Training Parameters:\\n{ppo_params}")\n'
call_line = "  _log_maxrl_scaffold_config(int(ppo_params.num_envs))\n"
install_line = "  _install_maxrl_loss_patch()\n"
if call_line not in text:
  if call_anchor in text:
    text = text.replace(call_anchor, call_anchor + call_line, 1)
  else:
    print("[bootstrap] MaxRL scaffold call anchor missing; skipping")
    raise SystemExit(0)

if install_line not in text:
  if call_line in text:
    text = text.replace(call_line, call_line + install_line, 1)
  else:
    print("[bootstrap] MaxRL install call anchor missing; skipping")
    raise SystemExit(0)

path.write_text(text)
print(f"[bootstrap] installed MaxRL scaffold shim at {path}")
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
echo "[bootstrap] numpy pin : ${NUMPY_VERSION:-<auto>}"
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
  apply_jax_clip_kwarg_compat_shim
  apply_maxrl_scaffold_shim
  VENV_PY="${VENV_DIR}/bin/python"
  if [ ! -x "${VENV_PY}" ]; then
    echo "ERROR: missing ${VENV_PY}. Re-run with BOOTSTRAP_OFFLINE=0 once."
    exit 1
  fi
  if [ "${USE_CUDA}" != "1" ]; then
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
    export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
  fi
  "${VENV_PY}" -B -c "import mujoco_playground; print('offline bootstrap ok')" >/dev/null
  echo "[bootstrap] offline validation passed"
  exit 0
fi

# Keep installs quota-friendly on shared HPC filesystems.
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_NO_COMPILE=1
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

VENV_PY="${VENV_DIR}/bin/python"
if [ ! -x "${VENV_PY}" ]; then
  echo "ERROR: venv python not found at ${VENV_PY}"
  exit 1
fi
export PYTHONPYCACHEPREFIX="${VENV_DIR}/.pycache"
echo "[bootstrap] venv py   : ${VENV_PY}"

VENV_SITE="$("${VENV_PY}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
echo "[bootstrap] venv site : ${VENV_SITE}"
case "${VENV_SITE}" in
  "${VENV_DIR}"/*) ;;
  *)
    echo "ERROR: venv site-packages is outside ${VENV_DIR}: ${VENV_SITE}"
    echo "Deactivate nested envs and re-run bootstrap from a clean shell."
    exit 1
    ;;
esac

# Prevent incompatible core pin sets from reaching a confusing pip failure.
"${VENV_PY}" -B - "${JAX_VERSION}" "${FLAX_VERSION}" <<'PY'
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

# Legacy jaxlib wheels (e.g., 0.4.x cuda11) are built against NumPy 1.x.
if [ -z "${NUMPY_VERSION}" ]; then
  NUMPY_VERSION="$("${VENV_PY}" -B - "${JAX_VERSION}" <<'PY'
import sys
def parse(v):
  parts = []
  for token in v.split("."):
    digits = "".join(ch for ch in token if ch.isdigit())
    parts.append(int(digits) if digits else 0)
  while len(parts) < 3:
    parts.append(0)
  return tuple(parts[:3])
print("1.26.4" if parse(sys.argv[1]) < (0, 5, 0) else "")
PY
)"
fi
echo "[bootstrap] numpy     : ${NUMPY_VERSION:-<none>}"

CONSTRAINTS_FILE="${VENV_DIR}/pip_constraints.txt"
cat > "${CONSTRAINTS_FILE}" <<EOF
jax==${JAX_VERSION}
jaxlib==${JAXLIB_VERSION}
flax==${FLAX_VERSION}
orbax-checkpoint==${ORBAX_VERSION}
EOF
if [ -n "${NUMPY_VERSION}" ]; then
  echo "numpy==${NUMPY_VERSION}" >> "${CONSTRAINTS_FILE}"
fi
if [ "${USE_CUDA}" = "1" ] && [ "${JAX_CUDA_EXTRA}" = "cuda11_pip" ]; then
  # jaxlib 0.4.25+cuda11.cudnn86 is linked against cuDNN 8.x.
  echo "nvidia-cudnn-cu11<9" >> "${CONSTRAINTS_FILE}"
fi
echo "[bootstrap] constraints: ${CONSTRAINTS_FILE}"

"${VENV_PY}" -B -m pip install "${PIP_FLAGS[@]}" --upgrade pip setuptools wheel

if [ ! -d "${PLAYGROUND_DIR}/.git" ]; then
  git "${GIT_ARGS[@]}" clone https://github.com/google-deepmind/mujoco_playground.git "${PLAYGROUND_DIR}"
fi

git_in_repo "${PLAYGROUND_DIR}" fetch --all --tags
git_in_repo "${PLAYGROUND_DIR}" checkout "${PLAYGROUND_REF}"
apply_mjx_make_data_compat_shim
apply_jax_clip_kwarg_compat_shim
apply_maxrl_scaffold_shim

if [ -n "${NUMPY_VERSION}" ]; then
  "${VENV_PY}" -B -m pip install "${PIP_FLAGS[@]}" --upgrade --prefer-binary -c "${CONSTRAINTS_FILE}" "numpy==${NUMPY_VERSION}"
fi
"${VENV_PY}" -B -m pip install "${PIP_FLAGS[@]}" --upgrade --prefer-binary --only-binary=ml_dtypes -c "${CONSTRAINTS_FILE}" "ml_dtypes==${ML_DTYPES_VERSION}"

if [ "${USE_CUDA}" = "1" ]; then
  if [ "${JAX_CUDA_EXTRA}" = "cuda11_pip" ]; then
    "${VENV_PY}" -B -m pip install "${PIP_FLAGS[@]}" --upgrade --prefer-binary --only-binary=ml_dtypes -c "${CONSTRAINTS_FILE}" \
      --find-links "${JAX_CUDA11_WHEELS_URL}" \
      "jax[${JAX_CUDA_EXTRA}]==${JAX_VERSION}" "jax==${JAX_VERSION}" "ml_dtypes==${ML_DTYPES_VERSION}"
  else
    "${VENV_PY}" -B -m pip install "${PIP_FLAGS[@]}" --upgrade --prefer-binary --only-binary=ml_dtypes -c "${CONSTRAINTS_FILE}" \
      "jax[${JAX_CUDA_EXTRA}]==${JAX_VERSION}" "jax==${JAX_VERSION}" "jaxlib==${JAXLIB_VERSION}" "ml_dtypes==${ML_DTYPES_VERSION}"
  fi
else
  "${VENV_PY}" -B -m pip install "${PIP_FLAGS[@]}" --upgrade --prefer-binary --only-binary=ml_dtypes -c "${CONSTRAINTS_FILE}" \
    "jax==${JAX_VERSION}" "jaxlib==${JAXLIB_VERSION}" "ml_dtypes==${ML_DTYPES_VERSION}"
fi

if [ "${PLAYGROUND_INSTALL_MODE}" = "full" ]; then
  "${VENV_PY}" -B -m pip install \
    "${PIP_FLAGS[@]}" \
    -c "${CONSTRAINTS_FILE}" \
    --prefer-binary \
    --only-binary=ml_dtypes \
    --extra-index-url https://py.mujoco.org \
    --extra-index-url https://pypi.nvidia.com \
    -e "${PLAYGROUND_DIR}[learning]"
else
  # Old enterprise Linux nodes often cannot install warp-lang/mujoco>=3.5 wheels.
  # This path installs a JAX-only stack for train_jax_ppo.py.
  "${VENV_PY}" -B -m pip install \
    "${PIP_FLAGS[@]}" \
    -c "${CONSTRAINTS_FILE}" \
    --prefer-binary \
    --only-binary=ml_dtypes \
    --extra-index-url https://py.mujoco.org \
    "mujoco==${MUJOCO_VERSION}" \
    "mujoco-mjx==${MUJOCO_MJX_VERSION}"

  "${VENV_PY}" -B -m pip install \
    "${PIP_FLAGS[@]}" \
    -c "${CONSTRAINTS_FILE}" \
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
    "${VENV_PY}" -B -m pip install "${PIP_FLAGS[@]}" --prefer-binary wandb
  fi

  "${VENV_PY}" -B -m pip install "${PIP_FLAGS[@]}" --no-deps -e "${PLAYGROUND_DIR}"
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
if ! git_in_repo "${MENAGERIE_DIR}" checkout "${MENAGERIE_COMMIT}"; then
  echo "[bootstrap] warning: menagerie commit ${MENAGERIE_COMMIT} not available in current clone."
  echo "[bootstrap] warning: attempting targeted fetch for that commit."
  git_in_repo "${MENAGERIE_DIR}" fetch origin "${MENAGERIE_COMMIT}" || true
  if ! git_in_repo "${MENAGERIE_DIR}" checkout "${MENAGERIE_COMMIT}"; then
    echo "[bootstrap] warning: unable to checkout ${MENAGERIE_COMMIT}; keeping current menagerie HEAD."
  fi
fi
echo "[bootstrap] menagerie ready at ${MENAGERIE_DIR}"

echo "[bootstrap] done"
echo "[bootstrap] activate with: source ${VENV_DIR}/bin/activate"
