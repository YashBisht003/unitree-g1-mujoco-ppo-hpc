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

apply_brax_checkpoint_restore_compat_shim() {
  local target
  target="$(find "${VENV_DIR}" -path '*/site-packages/brax/training/checkpoint.py' 2>/dev/null | head -n 1 || true)"
  if [ -z "${target}" ] || [ ! -f "${target}" ]; then
    echo "[bootstrap] warning: Brax checkpoint.py not found; skipping restore compat shim."
    return 0
  fi
  "${PYTHON_BIN}" - <<'PY' "${target}"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
marker = "__codex_brax_checkpoint_restore_v1__"

if marker in text:
  print("[bootstrap] Brax checkpoint restore compat shim already present")
  raise SystemExit(0)

old = """  metadata = ocp.PyTreeCheckpointer().metadata(path)\n  restore_args = jax.tree.map(\n      lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata\n  )\n"""
new = """  metadata = ocp.PyTreeCheckpointer().metadata(path)\n  metadata_tree = getattr(getattr(metadata, 'item_metadata', None), 'tree', metadata)\n  restore_args = jax.tree.map(\n      lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata_tree\n  )\n  # __codex_brax_checkpoint_restore_v1__\n"""

if old not in text:
  if "metadata_tree = getattr(getattr(metadata, 'item_metadata', None), 'tree', metadata)" in text:
    print("[bootstrap] Brax checkpoint restore compat shim already satisfied")
    raise SystemExit(0)
  print(f"[bootstrap] warning: restore compat anchor not found in {path}")
  raise SystemExit(0)

path.write_text(text.replace(old, new, 1))
print(f"[bootstrap] installed Brax checkpoint restore compat shim at {path}")
PY
}

apply_g1_recovery_reward_shim() {
  local target="${PLAYGROUND_DIR}/mujoco_playground/_src/locomotion/g1/joystick.py"
  if [ ! -f "${target}" ]; then
    echo "[bootstrap] warning: ${target} not found; skipping G1 recovery-reward shim."
    return 0
  fi

  "${PYTHON_BIN}" - "${target}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
marker = "__codex_g1_recovery_reward_v6__"
if marker in text:
  print("[bootstrap] G1 recovery-reward shim already present")
  raise SystemExit(0)

text = text.replace("# __codex_g1_recovery_reward_v3__\n", "")
text = text.replace("# __codex_g1_recovery_reward_v4__\n", "")
text = text.replace("# __codex_g1_recovery_reward_v5__\n", "")

if "class Joystick(g1_base.G1Env):" not in text:
  print("[bootstrap] G1 joystick layout not recognized; skipping recovery-reward shim")
  raise SystemExit(0)

if "recovery_ang_mom=1.0" not in text:
  anchor = "              tracking_ang_vel=0.75,\n"
  if anchor not in text:
    print("[bootstrap] failed to insert recovery scales; skipping")
    raise SystemExit(0)
  text = text.replace(
      anchor,
      anchor
      + "              recovery_ang_mom=1.0,\n"
      + "              recovery_bonus=1.0,\n",
      1,
  )

if "recovery_upright=1.0" not in text:
  anchor = "              recovery_bonus=1.0,\n"
  if anchor not in text:
    anchor = "              tracking_ang_vel=0.75,\n"
  if anchor not in text:
    print("[bootstrap] failed to insert force-adaptive recovery scales; skipping")
    raise SystemExit(0)
  text = text.replace(
      anchor,
      anchor
      + "              recovery_upright=1.0,\n"
      + "              recovery_com_vel=1.0,\n"
      + "              recovery_step=1.0,\n"
      + "              recovery_survival=1.0,\n",
      1,
  )

if "recovery_reward=config_dict.create(" not in text:
  recovery_block = """
      recovery_reward=config_dict.create(
          mode="off",  # off|recovery_window|force_adaptive
          window_steps=60,
          push_mask_window_steps=20,
          tracking_scale=0.3,
          tracking_scale_min=0.1,
          omega_weight=0.05,
          ang_mom_weight=1.0,
          ang_mom_severity_scale=0.5,
          ang_mom_sigma=1.5,
          upright_weight=1.5,
          upright_severity_scale=0.5,
          upright_sigma=0.2,
          com_weight=1.0,
          com_severity_scale=0.5,
          com_sigma=0.5,
          step_weight=0.8,
          step_air_time_min=0.15,
          survival_weight=0.25,
          bonus=4.0,
          bonus_severity_scale=4.0,
          bonus_stability_steps=10,
          bonus_delay_steps=10,
          stable_lin_tracking_min=0.7,
          stable_ang_tracking_min=0.7,
          capture_point_log=False,
      ),
"""
  text, n = re.subn(
      r'(\s+push_config=config_dict\.create\(\n\s+enable=True,\n\s+interval_range=\[[^\]]+\],\n\s+magnitude_range=\[[^\]]+\],\n\s+\),\n)',
      r"\1" + recovery_block,
      text,
      count=1,
  )
  if n == 0:
    print("[bootstrap] failed to insert recovery_reward config; skipping")
    raise SystemExit(0)

if "direction_frame=" not in text or "fixed_angle_deg=" not in text or "single_push=" not in text:
  push_block_pattern = re.compile(
      r'(\s+push_config=config_dict\.create\(\n\s+enable=True,\n\s+interval_range=\[[^\]]+\],\n\s+magnitude_range=\[[^\]]+\],\n)(\s+\),\n)'
  )
  push_block_insert = (
      r"\1"
      + '          direction_mode="uniform",\n'
      + '          direction_frame="world",  # world|body\n'
      + '          fixed_angle_deg=0.0,\n'
      + '          single_push=False,\n'
      + r"\2"
  )
  text, n = push_block_pattern.subn(push_block_insert, text, count=1)
  if n == 0:
    print("[bootstrap] failed to insert push direction config; skipping")
    raise SystemExit(0)

if "push_mask_window_steps=" not in text:
  text = text.replace(
      '          window_steps=60,\n',
      '          window_steps=60,\n'
      '          push_mask_window_steps=20,\n',
      1,
  )

text = text.replace(
    '          mode="off",  # off|recovery_window\n',
    '          mode="off",  # off|recovery_window|force_adaptive\n',
)

if "tracking_scale_min=" not in text:
  if '          tracking_scale=0.2,\n' in text:
    text = text.replace(
        '          tracking_scale=0.2,\n',
        '          tracking_scale=0.3,\n'
        '          tracking_scale_min=0.1,\n',
        1,
    )
  else:
    text = text.replace(
        '          tracking_scale=0.3,\n',
        '          tracking_scale=0.3,\n'
        '          tracking_scale_min=0.1,\n',
        1,
    )

if "ang_mom_weight=" not in text:
  text = text.replace(
      '          omega_weight=0.05,\n',
      '          omega_weight=0.05,\n'
      '          ang_mom_weight=1.0,\n'
      '          ang_mom_severity_scale=0.5,\n'
      '          ang_mom_sigma=1.5,\n',
      1,
  )

if "upright_weight=" not in text:
  text = text.replace(
      '          ang_mom_sigma=1.5,\n',
      '          ang_mom_sigma=1.5,\n'
      '          upright_weight=1.5,\n'
      '          upright_severity_scale=0.5,\n'
      '          upright_sigma=0.2,\n',
      1,
  )

if "com_weight=" not in text:
  text = text.replace(
      '          upright_sigma=0.2,\n',
      '          upright_sigma=0.2,\n'
      '          com_weight=1.0,\n'
      '          com_severity_scale=0.5,\n'
      '          com_sigma=0.5,\n',
      1,
  )

if "step_weight=" not in text:
  text = text.replace(
      '          com_sigma=0.5,\n',
      '          com_sigma=0.5,\n'
      '          step_weight=0.8,\n'
      '          step_air_time_min=0.15,\n'
      '          survival_weight=0.25,\n',
      1,
  )

if "bonus_severity_scale=" not in text:
  if '          bonus=8.0,\n' in text:
    text = text.replace(
        '          bonus=8.0,\n',
        '          bonus=4.0,\n'
        '          bonus_severity_scale=4.0,\n',
        1,
    )
  else:
    text = text.replace(
        '          bonus=4.0,\n',
        '          bonus=4.0,\n'
        '          bonus_severity_scale=4.0,\n',
        1,
    )

if '"recovery_countdown"' not in text:
  info_anchor = '        "push_interval_steps": push_interval_steps,\n'
  info_insert = (
      '        "push_interval_steps": push_interval_steps,\n'
      '        "recovery_countdown": jp.array(0, dtype=jp.int32),\n'
      '        "recovery_stable_steps": jp.array(0, dtype=jp.int32),\n'
      '        "recovery_survived": jp.array(1.0),\n'
      '        "recovery_bonus_flag": jp.array(0.0),\n'
      '        "in_recovery_window": jp.array(0.0),\n'
      '        "push_mask_countdown": jp.array(0, dtype=jp.int32),\n'
      '        "in_push_mask_window": jp.array(0.0),\n'
      '        "last_push_magnitude": jp.array(0.0),\n'
      '        "recovery_severity": jp.array(0.0),\n'
      '        "steps_since_push": jp.array(0, dtype=jp.int32),\n'
      '        "cp_pending_steps": jp.array(0, dtype=jp.int32),\n'
      '        "cp_valid": jp.array(0.0),\n'
      '        "cp_xy_norm": jp.array(0.0),\n'
      '        "cp_fail_window": jp.array(0.0),\n'
  )
  if info_anchor not in text:
    print("[bootstrap] failed to insert recovery info fields; skipping")
    raise SystemExit(0)
  text = text.replace(info_anchor, info_insert, 1)

if '"push_mask_countdown"' not in text:
  info_upgrade_anchor = '        "in_recovery_window": jp.array(0.0),\n'
  info_upgrade_insert = (
      '        "in_recovery_window": jp.array(0.0),\n'
      '        "push_mask_countdown": jp.array(0, dtype=jp.int32),\n'
      '        "in_push_mask_window": jp.array(0.0),\n'
      '        "last_push_magnitude": jp.array(0.0),\n'
      '        "recovery_severity": jp.array(0.0),\n'
      '        "steps_since_push": jp.array(0, dtype=jp.int32),\n'
  )
  if info_upgrade_anchor not in text:
    print("[bootstrap] failed to insert push-mask info fields; skipping")
    raise SystemExit(0)
  text = text.replace(info_upgrade_anchor, info_upgrade_insert, 1)

if '"last_push_theta"' not in text or '"push_count"' not in text:
  info_theta_anchor = '        "last_push_magnitude": jp.array(0.0),\n'
  info_theta_insert = (
      '        "last_push_magnitude": jp.array(0.0),\n'
      '        "last_push_theta": jp.array(0.0),\n'
      '        "push_count": jp.array(0, dtype=jp.int32),\n'
  )
  if info_theta_anchor not in text:
    print("[bootstrap] failed to insert push theta/count info fields; skipping")
    raise SystemExit(0)
  text = text.replace(info_theta_anchor, info_theta_insert, 1)

if '"last_push_magnitude"' not in text:
  info_upgrade_anchor = '        "in_push_mask_window": jp.array(0.0),\n'
  info_upgrade_insert = (
      '        "in_push_mask_window": jp.array(0.0),\n'
      '        "last_push_magnitude": jp.array(0.0),\n'
      '        "recovery_severity": jp.array(0.0),\n'
      '        "steps_since_push": jp.array(0, dtype=jp.int32),\n'
  )
  if info_upgrade_anchor not in text:
    print("[bootstrap] failed to insert recovery severity info fields; skipping")
    raise SystemExit(0)
  text = text.replace(info_upgrade_anchor, info_upgrade_insert, 1)

if 'metrics["diagnostics/recovery_countdown"] = jp.zeros(())' not in text:
  reset_metrics_anchor = '    metrics["swing_peak"] = jp.zeros(())\n'
  reset_metrics_insert = (
      '    metrics["swing_peak"] = jp.zeros(())\n'
      '    metrics["diagnostics/recovery_countdown"] = jp.zeros(())\n'
      '    metrics["diagnostics/in_recovery_window"] = jp.zeros(())\n'
      '    metrics["diagnostics/recovery_bonus_flag"] = jp.zeros(())\n'
      '    metrics["diagnostics/last_push_magnitude"] = jp.zeros(())\n'
      '    metrics["diagnostics/recovery_severity"] = jp.zeros(())\n'
      '    metrics["diagnostics/steps_since_push"] = jp.zeros(())\n'
      '    metrics["diagnostics/cp_valid"] = jp.zeros(())\n'
      '    metrics["diagnostics/cp_xy_norm"] = jp.zeros(())\n'
      '    metrics["diagnostics/cp_fail_window"] = jp.zeros(())\n'
  )
  if reset_metrics_anchor not in text:
    print("[bootstrap] failed to initialize recovery diagnostics metrics; skipping")
    raise SystemExit(0)
  text = text.replace(reset_metrics_anchor, reset_metrics_insert, 1)

if 'metrics["diagnostics/last_push_theta"] = jp.zeros(())' not in text:
  reset_theta_anchor = '    metrics["diagnostics/last_push_magnitude"] = jp.zeros(())\n'
  reset_theta_insert = (
      '    metrics["diagnostics/last_push_magnitude"] = jp.zeros(())\n'
      '    metrics["diagnostics/last_push_theta"] = jp.zeros(())\n'
  )
  if reset_theta_anchor not in text:
    print("[bootstrap] failed to initialize push theta metric; skipping")
    raise SystemExit(0)
  text = text.replace(reset_theta_anchor, reset_theta_insert, 1)

if 'if getattr(push_cfg, "direction_frame", "world") == "body":' not in text or 'if bool(getattr(push_cfg, "single_push", False)):' not in text:
  push_logic_pattern = re.compile(
      r'''    state\.info\["rng"\], push1_rng, push2_rng = jax\.random\.split\(\n        state\.info\["rng"\], 3\n    \)\n    push = jax\.random\.uniform\(push1_rng, minval=-1\.0, maxval=1\.0, shape=\(2,\)\)\n    push *= jp\.array\(self\._config\.push_config\.magnitude_range\)\n    push_trigger = \(\n        jp\.mod\(state\.info\["push_step"\] \+ 1, state\.info\["push_interval_steps"\]\) == 0\n    \)\n    push \*= push_trigger\n    push \*= self\._config\.push_config\.enable\n''',
      re.MULTILINE,
  )
  push_logic_repl = """    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_cfg = self._config.push_config
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    if getattr(push_cfg, "direction_mode", "uniform") == "fixed":
      push_theta = jp.deg2rad(
          jp.array(float(getattr(push_cfg, "fixed_angle_deg", 0.0)))
      )
    base_rot = math.quat_to_mat(state.data.qpos[3:7])
    base_yaw = jp.arctan2(base_rot[1, 0], base_rot[0, 0])
    if getattr(push_cfg, "direction_frame", "world") == "body":
      push_theta = push_theta + base_yaw
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=push_cfg.magnitude_range[0],
        maxval=push_cfg.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push_trigger = (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
    )
    if bool(getattr(push_cfg, "single_push", False)):
      push_trigger = push_trigger & (state.info["push_count"] < 1)
    push *= push_trigger
    push *= push_magnitude
    push *= push_cfg.enable
"""
  text, n = push_logic_pattern.subn(push_logic_repl, text, count=1)
  if n == 0:
    print("[bootstrap] failed to patch push direction logic; skipping")
    raise SystemExit(0)

if 'metrics["diagnostics/last_push_magnitude"] = jp.zeros(())' not in text:
  reset_metrics_anchor = '    metrics["diagnostics/recovery_bonus_flag"] = jp.zeros(())\n'
  reset_metrics_insert = (
      '    metrics["diagnostics/recovery_bonus_flag"] = jp.zeros(())\n'
      '    metrics["diagnostics/last_push_magnitude"] = jp.zeros(())\n'
      '    metrics["diagnostics/recovery_severity"] = jp.zeros(())\n'
      '    metrics["diagnostics/steps_since_push"] = jp.zeros(())\n'
  )
  if reset_metrics_anchor not in text:
    print("[bootstrap] failed to upgrade recovery diagnostics metrics; skipping")
    raise SystemExit(0)
  text = text.replace(reset_metrics_anchor, reset_metrics_insert, 1)

if "push_event = jp.linalg.norm(push) > 0" not in text:
  push_anchor = "    push *= self._config.push_config.enable\n"
  if push_anchor not in text:
    print("[bootstrap] failed to insert push_event marker; skipping")
    raise SystemExit(0)
  text = text.replace(
      push_anchor,
      push_anchor + "    push_event = jp.linalg.norm(push) > 0\n",
      1,
  )

step_pattern = re.compile(
    r'''    done = self\._get_termination\(data\)\n(?:[\s\S]*?)    rewards = self\._get_reward\(\n        data, action, state\.info, state\.metrics, done, first_contact, contact\n    \)\n''',
    re.MULTILINE,
)
new_block = """    done = self._get_termination(data)

    rec_cfg = self._config.recovery_reward
    recovery_enabled = rec_cfg.mode in ("recovery_window", "force_adaptive")
    window_steps = int(getattr(rec_cfg, "window_steps", 60))
    bonus_delay_steps = int(getattr(rec_cfg, "bonus_delay_steps", 10))
    bonus_stability_steps = int(getattr(rec_cfg, "bonus_stability_steps", 10))
    push_mask_window_steps = int(getattr(rec_cfg, "push_mask_window_steps", 20))
    stable_lin_min = float(getattr(rec_cfg, "stable_lin_tracking_min", 0.7))
    stable_ang_min = float(getattr(rec_cfg, "stable_ang_tracking_min", 0.7))
    upright_sigma = float(getattr(rec_cfg, "upright_sigma", 0.2))
    capture_point_log = bool(getattr(rec_cfg, "capture_point_log", False))
    total_recovery_steps = window_steps + bonus_delay_steps
    push_mag = jp.linalg.norm(push)
    push_min = jp.array(self._config.push_config.magnitude_range[0], dtype=push_mag.dtype)
    push_max = jp.array(self._config.push_config.magnitude_range[1], dtype=push_mag.dtype)
    push_span = jp.maximum(push_max - push_min, jp.array(1e-6, dtype=push_mag.dtype))
    push_severity_now = jp.clip((push_mag - push_min) / push_span, 0.0, 1.0)

    prev_countdown = state.info["recovery_countdown"]
    recovery_countdown = jp.where(
        recovery_enabled & push_event,
        total_recovery_steps,
        jp.maximum(prev_countdown - 1, 0),
    )
    in_recovery_window = recovery_enabled & (
        recovery_countdown > bonus_delay_steps
    )
    mask_window_steps = jp.maximum(
        1, jp.array(push_mask_window_steps, dtype=jp.int32)
    )
    push_mask_countdown = jp.where(
        push_event,
        mask_window_steps,
        jp.maximum(state.info["push_mask_countdown"] - 1, 0),
    )
    in_push_mask_window = push_mask_countdown > 0
    active_recovery_state = (recovery_countdown > 0) | (push_mask_countdown > 0)

    last_push_magnitude = jp.where(
        push_event, push_mag.astype(jp.float32), state.info["last_push_magnitude"]
    )
    recovery_severity = jp.where(
        push_event,
        push_severity_now.astype(jp.float32),
        jp.where(
            active_recovery_state,
            state.info["recovery_severity"],
            jp.array(0.0, dtype=jp.float32),
        ),
    )
    steps_since_push = jp.where(
        push_event,
        jp.array(0, dtype=jp.int32),
        jp.where(
            active_recovery_state,
            state.info["steps_since_push"] + 1,
            jp.array(0, dtype=jp.int32),
        ),
    )

    lin_track_for_bonus = self._reward_tracking_lin_vel(
        state.info["command"], self.get_local_linvel(data, "pelvis")
    )
    ang_track_for_bonus = self._reward_tracking_ang_vel(
        state.info["command"], self.get_gyro(data, "pelvis")
    )
    gravity_xy = self.get_gravity(data, "torso")[0:2]
    upright_for_bonus = jp.exp(
        -jp.square(
            jp.linalg.norm(gravity_xy)
            / jp.maximum(upright_sigma, jp.array(1e-3, dtype=gravity_xy.dtype))
        )
    )
    stable_step = (
        (lin_track_for_bonus >= stable_lin_min)
        & (ang_track_for_bonus >= stable_ang_min)
        & (upright_for_bonus >= 0.9)
        & (~done)
    )
    bonus_phase = recovery_enabled & (recovery_countdown > 0) & (
        recovery_countdown <= bonus_delay_steps
    )

    recovery_survived = jp.where(
        recovery_enabled & push_event,
        jp.array(1.0),
        state.info["recovery_survived"],
    )
    recovery_survived = jp.where(
        recovery_enabled & (recovery_countdown > 0),
        recovery_survived * (1.0 - done.astype(recovery_survived.dtype)),
        recovery_survived,
    )
    recovery_stable_steps = jp.where(
        recovery_enabled & push_event,
        jp.array(0, dtype=jp.int32),
        state.info["recovery_stable_steps"],
    )
    recovery_stable_steps = jp.where(
        bonus_phase & stable_step,
        recovery_stable_steps + 1,
        recovery_stable_steps,
    )
    recovery_bonus_flag = (
        recovery_enabled
        & (recovery_countdown == 1)
        & (recovery_survived > 0.5)
        & (recovery_stable_steps >= bonus_stability_steps)
    )

    cp_log_enabled = recovery_enabled & capture_point_log
    cp_omega0 = jp.sqrt(
        9.81 / jp.maximum(0.2, self._config.reward_config.base_height_target)
    )
    cp_xy = data.qpos[0:2] + data.qvel[0:2] / cp_omega0
    cp_norm = jp.linalg.norm(cp_xy)
    cp_pending = jp.where(
        cp_log_enabled & push_event,
        jp.array(5, dtype=jp.int32),
        state.info["cp_pending_steps"],
    )
    cp_capture_now = cp_log_enabled & (cp_pending == 1) & (~done)
    cp_valid = jp.where(
        cp_log_enabled & push_event,
        jp.array(0.0),
        state.info["cp_valid"],
    )
    cp_valid = jp.where(cp_capture_now, jp.array(1.0), cp_valid)
    cp_xy_norm = jp.where(cp_capture_now, cp_norm, state.info["cp_xy_norm"])
    cp_pending = jp.where(
        cp_log_enabled & (cp_pending > 0) & (~done), cp_pending - 1, cp_pending
    )
    cp_pending = jp.where(done, jp.array(0, dtype=jp.int32), cp_pending)
    cp_fail_window = jp.where(
        cp_log_enabled & done & (recovery_countdown > 0) & (cp_valid > 0.5),
        jp.array(1.0),
        jp.array(0.0),
    )

    state.info["recovery_countdown"] = recovery_countdown
    state.info["in_recovery_window"] = in_recovery_window.astype(jp.float32)
    state.info["push_mask_countdown"] = push_mask_countdown
    state.info["in_push_mask_window"] = in_push_mask_window.astype(jp.float32)
    state.info["last_push_magnitude"] = last_push_magnitude
    state.info["last_push_theta"] = jp.where(
        push_event, push_theta.astype(jp.float32), state.info["last_push_theta"]
    )
    state.info["recovery_severity"] = recovery_severity
    state.info["steps_since_push"] = steps_since_push
    state.info["recovery_survived"] = recovery_survived
    state.info["recovery_stable_steps"] = recovery_stable_steps
    state.info["recovery_bonus_flag"] = recovery_bonus_flag.astype(jp.float32)
    state.info["cp_pending_steps"] = cp_pending
    state.info["cp_valid"] = cp_valid
    state.info["cp_xy_norm"] = cp_xy_norm
    state.info["cp_fail_window"] = cp_fail_window
    state.info["push_count"] += push_event.astype(jp.int32)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
"""
text, step_replacements = step_pattern.subn(new_block, text, count=1)
if step_replacements == 0:
  print("[bootstrap] failed to patch step recovery logic; skipping")
  raise SystemExit(0)

reward_start = text.find("  def _get_reward(")
reward_end = text.find("  def _cost_contact_force(", reward_start)
if reward_start == -1 or reward_end == -1:
  print("[bootstrap] failed to locate _get_reward for patching; skipping")
  raise SystemExit(0)

new_reward_fn = '''
  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    rewards = {
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data, "pelvis")
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data, "pelvis")
        ),
        # Base-related rewards.
        "lin_vel_z": self._cost_lin_vel_z(
            self.get_global_linvel(data, "pelvis"),
            self.get_global_linvel(data, "torso"),
        ),
        "ang_vel_xy": self._cost_ang_vel_xy(
            self.get_global_angvel(data, "torso")
        ),
        "orientation": self._cost_orientation(self.get_gravity(data, "torso")),
        "base_height": self._cost_base_height(data.qpos[2]),
        # Energy related rewards.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        # Feet related rewards.
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, info),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_phase": self._reward_feet_phase(
            data,
            info["phase"],
            self._config.reward_config.max_foot_height,
            info["command"],
        ),
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "collision": self._cost_collision(data),
        "contact_force": self._cost_contact_force(data),
        # Pose related rewards.
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:], info["command"]
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "pose": self._cost_pose(data.qpos[7:]),
    }

    if self._config.recovery_reward.mode == "recovery_window":
      rec_cfg = self._config.recovery_reward
      in_window = info["in_recovery_window"] > 0.5
      track_scale = float(getattr(rec_cfg, "tracking_scale", 0.2))
      omega_weight = float(getattr(rec_cfg, "omega_weight", 0.05))
      bonus = float(getattr(rec_cfg, "bonus", 8.0))
      rewards["tracking_lin_vel"] = jp.where(
          in_window,
          rewards["tracking_lin_vel"] * track_scale,
          rewards["tracking_lin_vel"],
      )
      rewards["tracking_ang_vel"] = jp.where(
          in_window,
          rewards["tracking_ang_vel"] * track_scale,
          rewards["tracking_ang_vel"],
      )
      rewards["recovery_ang_mom"] = jp.where(
          in_window,
          -omega_weight * self._cost_ang_vel_xy(self.get_global_angvel(data, "torso")),
          jp.array(0.0),
      )
      rewards["recovery_bonus"] = bonus * info["recovery_bonus_flag"]
      rewards["recovery_upright"] = jp.array(0.0)
      rewards["recovery_com_vel"] = jp.array(0.0)
      rewards["recovery_step"] = jp.array(0.0)
      rewards["recovery_survival"] = jp.array(0.0)
    elif self._config.recovery_reward.mode == "force_adaptive":
      rec_cfg = self._config.recovery_reward
      in_window = info["in_recovery_window"] > 0.5
      severity = jp.clip(info["recovery_severity"], 0.0, 1.0)
      track_scale_max = float(getattr(rec_cfg, "tracking_scale", 0.3))
      track_scale_min = float(getattr(rec_cfg, "tracking_scale_min", 0.1))
      upright_weight = float(getattr(rec_cfg, "upright_weight", 1.5))
      upright_severity_scale = float(
          getattr(rec_cfg, "upright_severity_scale", 0.5)
      )
      upright_sigma = float(getattr(rec_cfg, "upright_sigma", 0.2))
      com_weight = float(getattr(rec_cfg, "com_weight", 1.0))
      com_severity_scale = float(getattr(rec_cfg, "com_severity_scale", 0.5))
      com_sigma = float(getattr(rec_cfg, "com_sigma", 0.5))
      ang_mom_weight = float(getattr(rec_cfg, "ang_mom_weight", 1.0))
      ang_mom_severity_scale = float(
          getattr(rec_cfg, "ang_mom_severity_scale", 0.5)
      )
      ang_mom_sigma = float(getattr(rec_cfg, "ang_mom_sigma", 1.5))
      step_weight = float(getattr(rec_cfg, "step_weight", 0.8))
      step_air_time_min = float(getattr(rec_cfg, "step_air_time_min", 0.15))
      survival_weight = float(getattr(rec_cfg, "survival_weight", 0.25))
      bonus = float(getattr(rec_cfg, "bonus", 4.0))
      bonus_severity_scale = float(getattr(rec_cfg, "bonus_severity_scale", 4.0))
      push_mask_window_steps = int(getattr(rec_cfg, "push_mask_window_steps", 20))
      tracking_scale = track_scale_max - severity * (
          track_scale_max - track_scale_min
      )
      tracking_scale = jp.clip(
          tracking_scale, track_scale_min, track_scale_max
      )
      rewards["tracking_lin_vel"] = jp.where(
          in_window,
          rewards["tracking_lin_vel"] * tracking_scale,
          rewards["tracking_lin_vel"],
      )
      rewards["tracking_ang_vel"] = jp.where(
          in_window,
          rewards["tracking_ang_vel"] * tracking_scale,
          rewards["tracking_ang_vel"],
      )

      gravity_xy = self.get_gravity(data, "torso")[0:2]
      upright = jp.exp(
          -jp.square(
              jp.linalg.norm(gravity_xy)
              / jp.maximum(upright_sigma, jp.array(1e-3, dtype=gravity_xy.dtype))
          )
      )
      local_linvel_xy = self.get_local_linvel(data, "pelvis")[0:2]
      command_xy = info["command"][0:2]
      com_vel = jp.exp(
          -jp.square(
              jp.linalg.norm(local_linvel_xy - command_xy)
              / jp.maximum(com_sigma, jp.array(1e-3, dtype=local_linvel_xy.dtype))
          )
      )
      ang_xy = self.get_global_angvel(data, "torso")[0:2]
      ang_mom = jp.exp(
          -jp.square(
              jp.linalg.norm(ang_xy)
              / jp.maximum(ang_mom_sigma, jp.array(1e-3, dtype=ang_xy.dtype))
          )
      )
      landing = first_contact.astype(jp.float32)
      air_time_ready = (
          info["feet_air_time"] >= step_air_time_min
      ).astype(jp.float32)
      step_active = in_window & (
          info["steps_since_push"] <= push_mask_window_steps
      )
      step_reward = jp.mean(landing * air_time_ready)

      rewards["recovery_ang_mom"] = jp.where(
          in_window,
          (ang_mom_weight + ang_mom_severity_scale * severity) * ang_mom,
          jp.array(0.0),
      )
      rewards["recovery_upright"] = jp.where(
          in_window,
          (upright_weight + upright_severity_scale * severity) * upright,
          jp.array(0.0),
      )
      rewards["recovery_com_vel"] = jp.where(
          in_window,
          (com_weight + com_severity_scale * severity) * com_vel,
          jp.array(0.0),
      )
      rewards["recovery_step"] = jp.where(
          step_active,
          step_weight * step_reward,
          jp.array(0.0),
      )
      rewards["recovery_survival"] = jp.where(
          in_window,
          survival_weight * (1.0 - done.astype(jp.float32)),
          jp.array(0.0),
      )
      rewards["recovery_bonus"] = (
          bonus + bonus_severity_scale * severity
      ) * info["recovery_bonus_flag"]
    else:
      rewards["recovery_ang_mom"] = jp.array(0.0)
      rewards["recovery_bonus"] = jp.array(0.0)
      rewards["recovery_upright"] = jp.array(0.0)
      rewards["recovery_com_vel"] = jp.array(0.0)
      rewards["recovery_step"] = jp.array(0.0)
      rewards["recovery_survival"] = jp.array(0.0)
    return rewards

'''
text = text[:reward_start] + new_reward_fn + text[reward_end:]

metrics_anchor = '''    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
'''
metrics_insert = '''    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
    state.metrics["diagnostics/recovery_countdown"] = state.info[
        "recovery_countdown"
    ].astype(reward.dtype)
    state.metrics["diagnostics/in_recovery_window"] = state.info[
        "in_recovery_window"
    ].astype(reward.dtype)
    state.metrics["diagnostics/recovery_bonus_flag"] = state.info[
        "recovery_bonus_flag"
    ].astype(reward.dtype)
    state.metrics["diagnostics/last_push_magnitude"] = state.info[
        "last_push_magnitude"
    ].astype(reward.dtype)
    state.metrics["diagnostics/last_push_theta"] = state.info[
        "last_push_theta"
    ].astype(reward.dtype)
    state.metrics["diagnostics/recovery_severity"] = state.info[
        "recovery_severity"
    ].astype(reward.dtype)
    state.metrics["diagnostics/steps_since_push"] = state.info[
        "steps_since_push"
    ].astype(reward.dtype)
    state.metrics["diagnostics/cp_valid"] = state.info["cp_valid"].astype(
        reward.dtype
    )
    state.metrics["diagnostics/cp_xy_norm"] = state.info["cp_xy_norm"].astype(
        reward.dtype
    )
    state.metrics["diagnostics/cp_fail_window"] = state.info[
        "cp_fail_window"
    ].astype(reward.dtype)
'''
if "diagnostics/recovery_countdown" not in text:
  if metrics_anchor not in text:
    print("[bootstrap] failed to insert recovery diagnostics metrics; skipping")
    raise SystemExit(0)
  text = text.replace(metrics_anchor, metrics_insert, 1)

if "diagnostics/last_push_magnitude" not in text:
  metrics_upgrade_anchor = '''    state.metrics["diagnostics/recovery_bonus_flag"] = state.info[
        "recovery_bonus_flag"
    ].astype(reward.dtype)
'''
  metrics_upgrade_insert = '''    state.metrics["diagnostics/recovery_bonus_flag"] = state.info[
        "recovery_bonus_flag"
    ].astype(reward.dtype)
    state.metrics["diagnostics/last_push_magnitude"] = state.info[
        "last_push_magnitude"
    ].astype(reward.dtype)
    state.metrics["diagnostics/last_push_theta"] = state.info[
        "last_push_theta"
    ].astype(reward.dtype)
    state.metrics["diagnostics/recovery_severity"] = state.info[
        "recovery_severity"
    ].astype(reward.dtype)
    state.metrics["diagnostics/steps_since_push"] = state.info[
        "steps_since_push"
    ].astype(reward.dtype)
'''
  if metrics_upgrade_anchor not in text:
    print("[bootstrap] failed to upgrade recovery diagnostics logging; skipping")
    raise SystemExit(0)
  text = text.replace(metrics_upgrade_anchor, metrics_upgrade_insert, 1)

text = text.replace(
    "class Joystick(g1_base.G1Env):",
    f"# {marker}\nclass Joystick(g1_base.G1Env):",
    1,
)

path.write_text(text)
print(f"[bootstrap] installed G1 recovery-reward shim at {path}")
PY
}

apply_g1_non_gait_recovery_shim() {
  local target="${PLAYGROUND_DIR}/mujoco_playground/_src/locomotion/g1/joystick.py"
  if [ ! -f "${target}" ]; then
    echo "[bootstrap] warning: ${target} not found; skipping G1 non-gait recovery shim."
    return 0
  fi

  "${PYTHON_BIN}" - "${target}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
marker = "__codex_g1_non_gait_recovery_v1__"
if marker in text:
  print("[bootstrap] G1 non-gait recovery shim already present")
  raise SystemExit(0)

if "biased_direction_prob=" not in text:
  anchor = '          direction_frame="world",  # world|body\n'
  if anchor not in text:
    print("[bootstrap] failed to insert biased push direction config; skipping")
    raise SystemExit(0)
  text = text.replace(
      anchor,
      anchor
      + '          biased_direction_prob=0.0,\n'
      + '          biased_direction_range_deg=[0.0, 360.0],\n',
      1,
  )

text = text.replace(
    '          mode="off",  # off|recovery_window|force_adaptive\n',
    '          mode="off",  # off|recovery_window|force_adaptive|recovery_gated\n',
)

if "gated_orientation_scale=" not in text:
  anchor = '          capture_point_log=False,\n'
  if anchor not in text:
    print("[bootstrap] failed to insert recovery_gated config; skipping")
    raise SystemExit(0)
  text = text.replace(
      anchor,
      anchor
      + '          gated_orientation_scale=0.2,\n'
      + '          gated_base_height_scale=0.5,\n',
      1,
  )

if 'biased_direction_prob' not in text or 'jax.random.bernoulli(push4_rng' not in text:
  old = """    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_cfg = self._config.push_config
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    if getattr(push_cfg, "direction_mode", "uniform") == "fixed":
      push_theta = jp.deg2rad(
          jp.array(float(getattr(push_cfg, "fixed_angle_deg", 0.0)))
      )
    base_rot = math.quat_to_mat(state.data.qpos[3:7])
    base_yaw = jp.arctan2(base_rot[1, 0], base_rot[0, 0])
    if getattr(push_cfg, "direction_frame", "world") == "body":
      push_theta = push_theta + base_yaw
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=push_cfg.magnitude_range[0],
        maxval=push_cfg.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push_trigger = (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
    )
    if bool(getattr(push_cfg, "single_push", False)):
      push_trigger = push_trigger & (state.info["push_count"] < 1)
    push *= push_trigger
    push *= push_magnitude
    push *= push_cfg.enable
"""
  new = """    state.info["rng"], push1_rng, push2_rng, push3_rng, push4_rng = jax.random.split(
        state.info["rng"], 5
    )
    push_cfg = self._config.push_config
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    if getattr(push_cfg, "direction_mode", "uniform") == "fixed":
      push_theta = jp.deg2rad(
          jp.array(float(getattr(push_cfg, "fixed_angle_deg", 0.0)))
      )
    else:
      biased_prob = float(getattr(push_cfg, "biased_direction_prob", 0.0))
      if biased_prob > 0.0:
        angle_range = getattr(
            push_cfg, "biased_direction_range_deg", [0.0, 360.0]
        )
        angle_min = jp.deg2rad(jp.array(float(angle_range[0]), dtype=jp.float32))
        angle_max = jp.deg2rad(jp.array(float(angle_range[1]), dtype=jp.float32))
        span = jp.mod(angle_max - angle_min, 2 * jp.pi)
        span = jp.where(span <= 0.0, 2 * jp.pi, span)
        biased_theta = jp.mod(
            angle_min + jax.random.uniform(push3_rng, maxval=span), 2 * jp.pi
        )
        use_biased = jax.random.bernoulli(push4_rng, p=biased_prob)
        push_theta = jp.where(use_biased, biased_theta, push_theta)
    base_rot = math.quat_to_mat(state.data.qpos[3:7])
    base_yaw = jp.arctan2(base_rot[1, 0], base_rot[0, 0])
    if getattr(push_cfg, "direction_frame", "world") == "body":
      push_theta = push_theta + base_yaw
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=push_cfg.magnitude_range[0],
        maxval=push_cfg.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push_trigger = (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
    )
    if bool(getattr(push_cfg, "single_push", False)):
      push_trigger = push_trigger & (state.info["push_count"] < 1)
    push *= push_trigger
    push *= push_magnitude
    push *= push_cfg.enable
"""
  if old not in text:
    print("[bootstrap] failed to patch biased push logic; skipping")
    raise SystemExit(0)
  text = text.replace(old, new, 1)

text = text.replace(
    '    recovery_enabled = rec_cfg.mode in ("recovery_window", "force_adaptive")\n',
    '    recovery_enabled = rec_cfg.mode in ("recovery_window", "force_adaptive", "recovery_gated")\n',
)

if 'elif self._config.recovery_reward.mode == "recovery_gated":' not in text:
  anchor = """      rewards["recovery_bonus"] = (
          bonus + bonus_severity_scale * severity
      ) * info["recovery_bonus_flag"]
    else:
"""
  insert = """      rewards["recovery_bonus"] = (
          bonus + bonus_severity_scale * severity
      ) * info["recovery_bonus_flag"]
    elif self._config.recovery_reward.mode == "recovery_gated":
      rec_cfg = self._config.recovery_reward
      in_window = info["in_recovery_window"] > 0.5
      orientation_scale = float(
          getattr(rec_cfg, "gated_orientation_scale", 0.2)
      )
      base_height_scale = float(
          getattr(rec_cfg, "gated_base_height_scale", 0.5)
      )
      gated_zero_keys = (
          "tracking_lin_vel",
          "tracking_ang_vel",
          "lin_vel_z",
          "ang_vel_xy",
          "torques",
          "energy",
          "dof_acc",
          "feet_slip",
          "feet_clearance",
          "feet_height",
          "feet_air_time",
          "feet_phase",
          "stand_still",
          "collision",
          "contact_force",
          "joint_deviation_hip",
          "joint_deviation_knee",
          "pose",
      )
      for key in gated_zero_keys:
        rewards[key] = jp.where(in_window, jp.array(0.0), rewards[key])
      rewards["orientation"] = jp.where(
          in_window,
          rewards["orientation"] * orientation_scale,
          rewards["orientation"],
      )
      rewards["base_height"] = jp.where(
          in_window,
          rewards["base_height"] * base_height_scale,
          rewards["base_height"],
      )
      rewards["recovery_ang_mom"] = jp.array(0.0)
      rewards["recovery_bonus"] = jp.array(0.0)
      rewards["recovery_upright"] = jp.array(0.0)
      rewards["recovery_com_vel"] = jp.array(0.0)
      rewards["recovery_step"] = jp.array(0.0)
      rewards["recovery_survival"] = jp.array(0.0)
    else:
"""
  if anchor not in text:
    print("[bootstrap] failed to insert recovery_gated reward branch; skipping")
    raise SystemExit(0)
  text = text.replace(anchor, insert, 1)

text = text.replace(
    "class Joystick(g1_base.G1Env):",
    f"# {marker}\nclass Joystick(g1_base.G1Env):",
    1,
)

path.write_text(text)
print(f"[bootstrap] installed G1 non-gait recovery shim at {path}")
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
marker = "__codex_maxrl_scaffold_v11__"
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
_PUSH_ADV_MASK_MODE = flags.DEFINE_enum(
    "push_adv_mask_mode",
    "off",
    ["off", "post_push_soft", "post_push_hard"],
    "Push-conditioned advantage masking mode.",
)
_PUSH_ADV_PRE_WEIGHT = flags.DEFINE_float(
    "push_adv_pre_weight",
    0.1,
    "Pre-push weight used by post_push_soft masking.",
)
_PUSH_MASK_SOURCE = flags.DEFINE_enum(
    "push_mask_source",
    "chunk",
    ["chunk", "stateful"],
    "Mask source: chunk-local cumsum on `push`, or stateful env window.",
)
_PUSH_MASK_WINDOW_K = flags.DEFINE_integer(
    "push_mask_window_k",
    20,
    "Stateful post-push mask window length in env steps.",
)
_PUSH_EVENT_EPS = flags.DEFINE_float(
    "push_event_eps",
    1e-6,
    "Threshold on |push| for detecting push timesteps.",
)
_PUSH_ENTROPY_MODE = flags.DEFINE_enum(
    "push_entropy_mode",
    "off",
    ["off", "post_push_additive"],
    "Push-conditioned entropy mode.",
)
_PUSH_ENTROPY_DELTA = flags.DEFINE_float(
    "push_entropy_delta",
    0.0,
    "Additional entropy coefficient on post-push timesteps (additive).",
)
_PUSH_REWARD_MODE = flags.DEFINE_enum(
    "push_reward_mode",
    "off",
    ["off", "recovery_window", "force_adaptive"],
    "Push-conditioned reward redesign mode.",
)
_RECOVERY_WINDOW_K = flags.DEFINE_integer(
    "recovery_window_k",
    60,
    "Recovery-window length in env steps.",
)
_RECOVERY_WINDOW_TRACKING_SCALE = flags.DEFINE_float(
    "recovery_window_tracking_scale",
    0.3,
    "Max tracking reward scale inside recovery window.",
)
_RECOVERY_WINDOW_TRACKING_SCALE_MIN = flags.DEFINE_float(
    "recovery_window_tracking_scale_min",
    0.1,
    "Min tracking reward scale at max push severity.",
)
_RECOVERY_OMEGA_WEIGHT = flags.DEFINE_float(
    "recovery_omega_weight",
    0.05,
    "Legacy recovery_window angular momentum regularization weight.",
)
_RECOVERY_ANG_MOM_WEIGHT = flags.DEFINE_float(
    "recovery_ang_mom_weight",
    1.0,
    "Base angular-momentum damping reward weight.",
)
_RECOVERY_ANG_MOM_SEVERITY_SCALE = flags.DEFINE_float(
    "recovery_ang_mom_severity_scale",
    0.5,
    "Additional angular-momentum damping weight at max push severity.",
)
_RECOVERY_ANG_MOM_SIGMA = flags.DEFINE_float(
    "recovery_ang_mom_sigma",
    1.5,
    "Angular-velocity sigma for force-adaptive damping reward.",
)
_RECOVERY_UPRIGHT_WEIGHT = flags.DEFINE_float(
    "recovery_upright_weight",
    1.5,
    "Base upright reward weight inside recovery window.",
)
_RECOVERY_UPRIGHT_SEVERITY_SCALE = flags.DEFINE_float(
    "recovery_upright_severity_scale",
    0.5,
    "Additional upright reward weight at max push severity.",
)
_RECOVERY_UPRIGHT_SIGMA = flags.DEFINE_float(
    "recovery_upright_sigma",
    0.2,
    "Tilt sigma for force-adaptive upright reward.",
)
_RECOVERY_COM_WEIGHT = flags.DEFINE_float(
    "recovery_com_weight",
    1.0,
    "Base COM velocity return reward weight.",
)
_RECOVERY_COM_SEVERITY_SCALE = flags.DEFINE_float(
    "recovery_com_severity_scale",
    0.5,
    "Additional COM velocity return reward weight at max push severity.",
)
_RECOVERY_COM_SIGMA = flags.DEFINE_float(
    "recovery_com_sigma",
    0.5,
    "Velocity-error sigma for COM return reward.",
)
_RECOVERY_STEP_WEIGHT = flags.DEFINE_float(
    "recovery_step_weight",
    0.8,
    "Landing reward weight for recovery steps shortly after a push.",
)
_RECOVERY_STEP_AIR_TIME_MIN = flags.DEFINE_float(
    "recovery_step_air_time_min",
    0.15,
    "Minimum foot air-time before a landing counts as a recovery step.",
)
_RECOVERY_SURVIVAL_WEIGHT = flags.DEFINE_float(
    "recovery_survival_weight",
    0.25,
    "Per-step survival reward inside the recovery window.",
)
_RECOVERY_BONUS = flags.DEFINE_float(
    "recovery_bonus",
    4.0,
    "Sparse bonus for stable recovery after window end.",
)
_RECOVERY_BONUS_SEVERITY_SCALE = flags.DEFINE_float(
    "recovery_bonus_severity_scale",
    4.0,
    "Additional stable-recovery bonus at max push severity.",
)
_RECOVERY_BONUS_STABILITY_STEPS = flags.DEFINE_integer(
    "recovery_bonus_stability_steps",
    10,
    "Required stable steps before recovery bonus can trigger.",
)
_RECOVERY_BONUS_DELAY_STEPS = flags.DEFINE_integer(
    "recovery_bonus_delay_steps",
    10,
    "Delay after recovery window before bonus trigger check.",
)
_RECOVERY_STABLE_LIN_MIN = flags.DEFINE_float(
    "recovery_stable_lin_min",
    0.7,
    "Minimum linear tracking score for a step to count as stable.",
)
_RECOVERY_STABLE_ANG_MIN = flags.DEFINE_float(
    "recovery_stable_ang_min",
    0.7,
    "Minimum angular tracking score for a step to count as stable.",
)
_CAPTURE_POINT_LOG = flags.DEFINE_boolean(
    "capture_point_log",
    False,
    "Enable capture-point proxy logging diagnostics.",
)
"""
flags_block = flags_block.replace("__MARKER__", marker)

helpers_block = """
# __MARKER__
def _merge_overrides(dst, src):
  out = dict(dst)
  for k, v in src.items():
    if isinstance(v, dict) and isinstance(out.get(k), dict):
      out[k] = _merge_overrides(out[k], v)
    else:
      out[k] = v
  return out


def _build_recovery_reward_overrides():
  overrides = {}
  if _PUSH_REWARD_MODE.value != "off":
    overrides = {
        "recovery_reward": {
            "mode": _PUSH_REWARD_MODE.value,
            "window_steps": int(_RECOVERY_WINDOW_K.value),
            "tracking_scale": float(_RECOVERY_WINDOW_TRACKING_SCALE.value),
            "tracking_scale_min": float(_RECOVERY_WINDOW_TRACKING_SCALE_MIN.value),
            "omega_weight": float(_RECOVERY_OMEGA_WEIGHT.value),
            "ang_mom_weight": float(_RECOVERY_ANG_MOM_WEIGHT.value),
            "ang_mom_severity_scale": float(
                _RECOVERY_ANG_MOM_SEVERITY_SCALE.value
            ),
            "ang_mom_sigma": float(_RECOVERY_ANG_MOM_SIGMA.value),
            "upright_weight": float(_RECOVERY_UPRIGHT_WEIGHT.value),
            "upright_severity_scale": float(
                _RECOVERY_UPRIGHT_SEVERITY_SCALE.value
            ),
            "upright_sigma": float(_RECOVERY_UPRIGHT_SIGMA.value),
            "com_weight": float(_RECOVERY_COM_WEIGHT.value),
            "com_severity_scale": float(_RECOVERY_COM_SEVERITY_SCALE.value),
            "com_sigma": float(_RECOVERY_COM_SIGMA.value),
            "step_weight": float(_RECOVERY_STEP_WEIGHT.value),
            "step_air_time_min": float(_RECOVERY_STEP_AIR_TIME_MIN.value),
            "survival_weight": float(_RECOVERY_SURVIVAL_WEIGHT.value),
            "bonus": float(_RECOVERY_BONUS.value),
            "bonus_severity_scale": float(_RECOVERY_BONUS_SEVERITY_SCALE.value),
            "bonus_stability_steps": int(_RECOVERY_BONUS_STABILITY_STEPS.value),
            "bonus_delay_steps": int(_RECOVERY_BONUS_DELAY_STEPS.value),
            "stable_lin_tracking_min": float(_RECOVERY_STABLE_LIN_MIN.value),
            "stable_ang_tracking_min": float(_RECOVERY_STABLE_ANG_MIN.value),
            "capture_point_log": bool(_CAPTURE_POINT_LOG.value),
        }
    }

  if _PUSH_ADV_MASK_MODE.value != "off" and _PUSH_MASK_SOURCE.value == "stateful":
    overrides = _merge_overrides(
        overrides,
        {
            "recovery_reward": {
                "push_mask_window_steps": int(_PUSH_MASK_WINDOW_K.value),
            }
        },
    )
  return overrides


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


def _post_push_mask_from_state_extras(state_extras, dtype):
  \"\"\"Returns post-push mask [T, N], or None when push signal is unavailable.\"\"\"

  if _PUSH_MASK_SOURCE.value == "stateful":
    mask_window = (
        state_extras["in_push_mask_window"]
        if "in_push_mask_window" in state_extras
        else None
    )
    if mask_window is not None:
      return (mask_window > 0.5).astype(dtype)

  push_vec = state_extras["push"] if "push" in state_extras else None
  if push_vec is None:
    return None
  push_mag = jp.linalg.norm(push_vec, axis=-1)
  push_event = (push_mag > _PUSH_EVENT_EPS.value).astype(dtype)
  return (jp.cumsum(push_event, axis=0) > 0).astype(dtype)


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
  state_extras = data.extras["state_extras"]
  truncation = state_extras["truncation"]
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
  push_mask_mean_weight = jp.array(1.0, dtype=rewards.dtype)
  push_mask_post_fraction = jp.array(0.0, dtype=rewards.dtype)
  push_entropy_mean_coeff = jp.array(entropy_cost, dtype=rewards.dtype)
  adv_mode = _ADV_MODE.value
  group_size = int(_SCENARIO_GROUP_SIZE.value)

  if adv_mode == "maxrl_binary" and not _MAXRL_LOG_ONLY.value:
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

  push_mask_mode = _PUSH_ADV_MASK_MODE.value
  post_push = None
  if push_mask_mode != "off":
    post_push = _post_push_mask_from_state_extras(state_extras, advantages.dtype)
    if post_push is None:
      raise ValueError(
          "push_adv_mask_mode requires state_extras['push'], but it is missing. "
          "Ensure _install_push_mask_patch is active and env exposes state.info['push']."
      )
    if push_mask_mode == "post_push_soft":
      pre_w = _PUSH_ADV_PRE_WEIGHT.value
      adv_mask = pre_w + (1.0 - pre_w) * post_push
    else:
      adv_mask = post_push
    # If a rollout chunk has no push event, keep PPO-style updates for it.
    has_push = jp.max(post_push, axis=0, keepdims=True)
    adv_mask = jp.where(has_push > 0, adv_mask, jp.ones_like(adv_mask))
    advantages = advantages * adv_mask
    push_mask_mean_weight = jp.mean(adv_mask)
    push_mask_post_fraction = jp.mean(post_push)

  rho_s = jp.exp(target_action_log_probs - behaviour_action_log_probs)
  surrogate_loss1 = rho_s * advantages
  surrogate_loss2 = jp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
  policy_loss = -jp.mean(jp.minimum(surrogate_loss1, surrogate_loss2))

  v_error = vs - baseline
  v_loss = jp.mean(v_error * v_error) * 0.5 * 0.5

  entropy_per_t = parametric_action_distribution.entropy(policy_logits, rng)
  push_entropy_mode = _PUSH_ENTROPY_MODE.value
  if push_entropy_mode != "off":
    if post_push is None:
      post_push = _post_push_mask_from_state_extras(state_extras, advantages.dtype)
    if post_push is None:
      raise ValueError(
          "push_entropy_mode requires state_extras['push'], but it is missing. "
          "Ensure _install_push_mask_patch is active and env exposes state.info['push']."
      )
    if push_entropy_mode == "post_push_additive":
      entropy_coeff = entropy_cost + _PUSH_ENTROPY_DELTA.value * post_push
    else:
      entropy_coeff = jp.ones_like(post_push) * entropy_cost
    entropy_loss = -jp.mean(entropy_coeff * entropy_per_t)
    push_entropy_mean_coeff = jp.mean(entropy_coeff)
  else:
    entropy_loss = entropy_cost * -jp.mean(entropy_per_t)

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
  if _PUSH_ADV_MASK_MODE.value != "off":
    metrics["push_mask/mean_weight"] = push_mask_mean_weight
    metrics["push_mask/post_fraction"] = push_mask_post_fraction
  if _PUSH_ENTROPY_MODE.value != "off":
    metrics["push_entropy/mean_coeff"] = push_entropy_mean_coeff
  return total_loss, metrics


def _install_push_mask_patch() -> None:
  \"\"\"Ensures `push` is available in state_extras when push-conditioned modes are enabled.\"\"\"

  mask_mode = _PUSH_ADV_MASK_MODE.value
  entropy_mode = _PUSH_ENTROPY_MODE.value
  if mask_mode == "off" and entropy_mode == "off":
    return

  from brax.training import acting as brax_acting

  current = brax_acting.generate_unroll
  if getattr(current, "__name__", "") == "_generate_unroll_with_push":
    return

  def _generate_unroll_with_push(
      env, env_state, policy, key, unroll_length, extra_fields=()
  ):
    extras = tuple(extra_fields) if extra_fields is not None else ()
    if "push" not in extras:
      extras = extras + ("push",)
    if _PUSH_MASK_SOURCE.value == "stateful" and "in_push_mask_window" not in extras:
      extras = extras + ("in_push_mask_window",)
    return current(
        env, env_state, policy, key, unroll_length, extra_fields=extras
    )

  brax_acting.generate_unroll = _generate_unroll_with_push
  print(
      "[push] requested `push` in state_extras "
      f"(mask_mode={mask_mode}, entropy_mode={entropy_mode}, "
      f"mask_source={_PUSH_MASK_SOURCE.value}, mask_k={_PUSH_MASK_WINDOW_K.value}, "
      f"pre_weight={_PUSH_ADV_PRE_WEIGHT.value}, entropy_delta={_PUSH_ENTROPY_DELTA.value})."
  )


def _install_maxrl_loss_patch() -> None:
  \"\"\"Installs runtime PPO-loss patch for MaxRL and push-conditioned objectives.\"\"\"

  adv_mode = _ADV_MODE.value
  need_push_objective = (
      _PUSH_ADV_MASK_MODE.value != "off" or _PUSH_ENTROPY_MODE.value != "off"
  )
  need_maxrl_objective = adv_mode != "ppo" and not _MAXRL_LOG_ONLY.value
  if not need_push_objective and not need_maxrl_objective:
    if adv_mode != "ppo" and _MAXRL_LOG_ONLY.value:
      print("[maxrl] maxrl_log_only=True -> keeping baseline PPO loss.")
    return

  patched_name = getattr(ppo_losses.compute_ppo_loss, "__name__", "")
  if patched_name == "_compute_ppo_loss_with_maxrl":
    return

  ppo_losses.compute_ppo_loss = _compute_ppo_loss_with_maxrl
  print(
      "[objective] installed PPO loss patch "
      f"(adv_mode={adv_mode}, push_adv_mask_mode={_PUSH_ADV_MASK_MODE.value}, "
      f"push_entropy_mode={_PUSH_ENTROPY_MODE.value}, "
      f"scenario_group_size={_SCENARIO_GROUP_SIZE.value})."
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

  if _PUSH_ADV_MASK_MODE.value != "off":
    print(
        "[push-mask] mode="
        f"{_PUSH_ADV_MASK_MODE.value}, pre_weight={_PUSH_ADV_PRE_WEIGHT.value}, "
        f"source={_PUSH_MASK_SOURCE.value}, K={_PUSH_MASK_WINDOW_K.value}, "
        f"event_eps={_PUSH_EVENT_EPS.value}"
    )
  if _PUSH_ENTROPY_MODE.value != "off":
    print(
        "[push-entropy] mode="
        f"{_PUSH_ENTROPY_MODE.value}, delta={_PUSH_ENTROPY_DELTA.value}, "
        f"event_eps={_PUSH_EVENT_EPS.value}"
    )
  if _PUSH_REWARD_MODE.value != "off":
    print(
        "[push-reward] mode="
        f"{_PUSH_REWARD_MODE.value}, K={_RECOVERY_WINDOW_K.value}, "
        f"tracking_scale=[{_RECOVERY_WINDOW_TRACKING_SCALE_MIN.value},"
        f"{_RECOVERY_WINDOW_TRACKING_SCALE.value}], "
        f"ang_mom_w={_RECOVERY_ANG_MOM_WEIGHT.value}, "
        f"survival_w={_RECOVERY_SURVIVAL_WEIGHT.value}, "
        f"bonus={_RECOVERY_BONUS.value}+{_RECOVERY_BONUS_SEVERITY_SCALE.value}*s, "
        f"stability_steps={_RECOVERY_BONUS_STABILITY_STEPS.value}, "
        f"bonus_delay={_RECOVERY_BONUS_DELAY_STEPS.value}, "
        f"cp_log={_CAPTURE_POINT_LOG.value}"
    )

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

if "_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(" not in text:
  playground_overrides_flag = """
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    "playground_config_overrides",
    None,
    "Overrides for the playground env config.",
)
"""
  text, n = re.subn(
      r'(_IMPL\s*=\s*flags\.DEFINE_enum\([\s\S]*?\)\n)',
      r"\1" + playground_overrides_flag + "\n",
      text,
      count=1,
  )
  if n == 0:
    text, n2 = re.subn(
        r"\n\ndef get_rl_config\(",
        "\n" + playground_overrides_flag + "\n\ndef get_rl_config(",
        text,
        count=1,
    )
    if n2 == 0:
      print("[bootstrap] playground_config_overrides flag insertion failed; skipping")
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
    # Fallback anchor for pre-existing scaffold layouts.
    text, n2 = re.subn(
        r'(_MAXRL_VERBOSE\s*=\s*flags\.DEFINE_boolean\([\s\S]*?\)\n)',
        r"\1" + episode_flag_block + "\n",
        text,
        count=1,
    )
    if n2 == 0:
      print("[bootstrap] MaxRL episode verifier flag insertion failed; skipping")
      raise SystemExit(0)

if "_PUSH_ADV_MASK_MODE = flags.DEFINE_enum(" not in text:
  push_mask_mode_block = """
_PUSH_ADV_MASK_MODE = flags.DEFINE_enum(
    "push_adv_mask_mode",
    "off",
    ["off", "post_push_soft", "post_push_hard"],
    "Push-conditioned advantage masking mode.",
)
"""
  text, n = re.subn(
      r"\n\ndef get_rl_config\(",
      "\n" + push_mask_mode_block + "\n\ndef get_rl_config(",
      text,
      count=1,
  )
  if n == 0:
    text, n2 = re.subn(
        r'(_MAXRL_EPISODE_VERIFIER\s*=\s*flags\.DEFINE_boolean\([\s\S]*?\)\n)',
        r"\1" + push_mask_mode_block + "\n",
        text,
        count=1,
    )
    if n2 == 0:
      print("[bootstrap] push_adv_mask_mode flag insertion failed; skipping")
      raise SystemExit(0)

if "_PUSH_ADV_PRE_WEIGHT = flags.DEFINE_float(" not in text:
  push_pre_weight_block = """
_PUSH_ADV_PRE_WEIGHT = flags.DEFINE_float(
    "push_adv_pre_weight",
    0.1,
    "Pre-push weight used by post_push_soft masking.",
)
"""
  text, n = re.subn(
      r"\n\ndef get_rl_config\(",
      "\n" + push_pre_weight_block + "\n\ndef get_rl_config(",
      text,
      count=1,
  )
  if n == 0:
    text, n2 = re.subn(
        r'(_PUSH_ADV_MASK_MODE\s*=\s*flags\.DEFINE_enum\([\s\S]*?\)\n)',
        r"\1" + push_pre_weight_block + "\n",
        text,
        count=1,
    )
    if n2 == 0:
      print("[bootstrap] push_adv_pre_weight flag insertion failed; skipping")
      raise SystemExit(0)

if "_PUSH_MASK_SOURCE = flags.DEFINE_enum(" not in text:
  push_mask_source_block = """
_PUSH_MASK_SOURCE = flags.DEFINE_enum(
    "push_mask_source",
    "chunk",
    ["chunk", "stateful"],
    "Mask source: chunk-local cumsum on `push`, or stateful env window.",
)
"""
  text, n = re.subn(
      r"\n\ndef get_rl_config\(",
      "\n" + push_mask_source_block + "\n\ndef get_rl_config(",
      text,
      count=1,
  )
  if n == 0:
    text, n2 = re.subn(
        r'(_PUSH_ADV_PRE_WEIGHT\s*=\s*flags\.DEFINE_float\([\s\S]*?\)\n)',
        r"\1" + push_mask_source_block + "\n",
        text,
        count=1,
    )
    if n2 == 0:
      print("[bootstrap] push_mask_source flag insertion failed; skipping")
      raise SystemExit(0)

if "_PUSH_MASK_WINDOW_K = flags.DEFINE_integer(" not in text:
  push_mask_k_block = """
_PUSH_MASK_WINDOW_K = flags.DEFINE_integer(
    "push_mask_window_k",
    20,
    "Stateful post-push mask window length in env steps.",
)
"""
  text, n = re.subn(
      r"\n\ndef get_rl_config\(",
      "\n" + push_mask_k_block + "\n\ndef get_rl_config(",
      text,
      count=1,
  )
  if n == 0:
    text, n2 = re.subn(
        r'(_PUSH_MASK_SOURCE\s*=\s*flags\.DEFINE_enum\([\s\S]*?\)\n)',
        r"\1" + push_mask_k_block + "\n",
        text,
        count=1,
    )
    if n2 == 0:
      print("[bootstrap] push_mask_window_k flag insertion failed; skipping")
      raise SystemExit(0)

if "_PUSH_EVENT_EPS = flags.DEFINE_float(" not in text:
  push_event_eps_block = """
_PUSH_EVENT_EPS = flags.DEFINE_float(
    "push_event_eps",
    1e-6,
    "Threshold on |push| for detecting push timesteps.",
)
"""
  text, n = re.subn(
      r"\n\ndef get_rl_config\(",
      "\n" + push_event_eps_block + "\n\ndef get_rl_config(",
      text,
      count=1,
  )
  if n == 0:
    text, n2 = re.subn(
        r'(_PUSH_ADV_PRE_WEIGHT\s*=\s*flags\.DEFINE_float\([\s\S]*?\)\n)',
        r"\1" + push_event_eps_block + "\n",
        text,
        count=1,
    )
    if n2 == 0:
      print("[bootstrap] push_event_eps flag insertion failed; skipping")
      raise SystemExit(0)

if "_PUSH_ENTROPY_MODE = flags.DEFINE_enum(" not in text:
  push_entropy_mode_block = """
_PUSH_ENTROPY_MODE = flags.DEFINE_enum(
    "push_entropy_mode",
    "off",
    ["off", "post_push_additive"],
    "Push-conditioned entropy mode.",
)
"""
  text, n = re.subn(
      r"\n\ndef get_rl_config\(",
      "\n" + push_entropy_mode_block + "\n\ndef get_rl_config(",
      text,
      count=1,
  )
  if n == 0:
    text, n2 = re.subn(
        r'(_PUSH_EVENT_EPS\s*=\s*flags\.DEFINE_float\([\s\S]*?\)\n)',
        r"\1" + push_entropy_mode_block + "\n",
        text,
        count=1,
    )
    if n2 == 0:
      print("[bootstrap] push_entropy_mode flag insertion failed; skipping")
      raise SystemExit(0)

if "_PUSH_ENTROPY_DELTA = flags.DEFINE_float(" not in text:
  push_entropy_delta_block = """
_PUSH_ENTROPY_DELTA = flags.DEFINE_float(
    "push_entropy_delta",
    0.0,
    "Additional entropy coefficient on post-push timesteps (additive).",
)
"""
  text, n = re.subn(
      r"\n\ndef get_rl_config\(",
      "\n" + push_entropy_delta_block + "\n\ndef get_rl_config(",
      text,
      count=1,
  )
  if n == 0:
    text, n2 = re.subn(
        r'(_PUSH_ENTROPY_MODE\s*=\s*flags\.DEFINE_enum\([\s\S]*?\)\n)',
        r"\1" + push_entropy_delta_block + "\n",
        text,
        count=1,
    )
    if n2 == 0:
      print("[bootstrap] push_entropy_delta flag insertion failed; skipping")
      raise SystemExit(0)

if "_PUSH_REWARD_MODE = flags.DEFINE_enum(" not in text:
  recovery_reward_flags_block = """
_PUSH_REWARD_MODE = flags.DEFINE_enum(
    "push_reward_mode",
    "off",
    ["off", "recovery_window", "force_adaptive"],
    "Push-conditioned reward redesign mode.",
)
_RECOVERY_WINDOW_K = flags.DEFINE_integer(
    "recovery_window_k",
    60,
    "Recovery-window length in env steps.",
)
_RECOVERY_WINDOW_TRACKING_SCALE = flags.DEFINE_float(
    "recovery_window_tracking_scale",
    0.3,
    "Max tracking reward scale inside recovery window.",
)
_RECOVERY_WINDOW_TRACKING_SCALE_MIN = flags.DEFINE_float(
    "recovery_window_tracking_scale_min",
    0.1,
    "Min tracking reward scale at max push severity.",
)
_RECOVERY_OMEGA_WEIGHT = flags.DEFINE_float(
    "recovery_omega_weight",
    0.05,
    "Legacy recovery_window angular momentum regularization weight.",
)
_RECOVERY_ANG_MOM_WEIGHT = flags.DEFINE_float(
    "recovery_ang_mom_weight",
    1.0,
    "Base angular-momentum damping reward weight.",
)
_RECOVERY_ANG_MOM_SEVERITY_SCALE = flags.DEFINE_float(
    "recovery_ang_mom_severity_scale",
    0.5,
    "Additional angular-momentum damping weight at max push severity.",
)
_RECOVERY_ANG_MOM_SIGMA = flags.DEFINE_float(
    "recovery_ang_mom_sigma",
    1.5,
    "Angular-velocity sigma for force-adaptive damping reward.",
)
_RECOVERY_UPRIGHT_WEIGHT = flags.DEFINE_float(
    "recovery_upright_weight",
    1.5,
    "Base upright reward weight inside recovery window.",
)
_RECOVERY_UPRIGHT_SEVERITY_SCALE = flags.DEFINE_float(
    "recovery_upright_severity_scale",
    0.5,
    "Additional upright reward weight at max push severity.",
)
_RECOVERY_UPRIGHT_SIGMA = flags.DEFINE_float(
    "recovery_upright_sigma",
    0.2,
    "Tilt sigma for force-adaptive upright reward.",
)
_RECOVERY_COM_WEIGHT = flags.DEFINE_float(
    "recovery_com_weight",
    1.0,
    "Base COM velocity return reward weight.",
)
_RECOVERY_COM_SEVERITY_SCALE = flags.DEFINE_float(
    "recovery_com_severity_scale",
    0.5,
    "Additional COM velocity return reward weight at max push severity.",
)
_RECOVERY_COM_SIGMA = flags.DEFINE_float(
    "recovery_com_sigma",
    0.5,
    "Velocity-error sigma for COM return reward.",
)
_RECOVERY_STEP_WEIGHT = flags.DEFINE_float(
    "recovery_step_weight",
    0.8,
    "Landing reward weight for recovery steps shortly after a push.",
)
_RECOVERY_STEP_AIR_TIME_MIN = flags.DEFINE_float(
    "recovery_step_air_time_min",
    0.15,
    "Minimum foot air-time before a landing counts as a recovery step.",
)
_RECOVERY_SURVIVAL_WEIGHT = flags.DEFINE_float(
    "recovery_survival_weight",
    0.25,
    "Per-step survival reward inside the recovery window.",
)
_RECOVERY_BONUS = flags.DEFINE_float(
    "recovery_bonus",
    4.0,
    "Sparse bonus for stable recovery after window end.",
)
_RECOVERY_BONUS_SEVERITY_SCALE = flags.DEFINE_float(
    "recovery_bonus_severity_scale",
    4.0,
    "Additional stable-recovery bonus at max push severity.",
)
_RECOVERY_BONUS_STABILITY_STEPS = flags.DEFINE_integer(
    "recovery_bonus_stability_steps",
    10,
    "Required stable steps before recovery bonus can trigger.",
)
_RECOVERY_BONUS_DELAY_STEPS = flags.DEFINE_integer(
    "recovery_bonus_delay_steps",
    10,
    "Delay after recovery window before bonus trigger check.",
)
_RECOVERY_STABLE_LIN_MIN = flags.DEFINE_float(
    "recovery_stable_lin_min",
    0.7,
    "Minimum linear tracking score for a step to count as stable.",
)
_RECOVERY_STABLE_ANG_MIN = flags.DEFINE_float(
    "recovery_stable_ang_min",
    0.7,
    "Minimum angular tracking score for a step to count as stable.",
)
_CAPTURE_POINT_LOG = flags.DEFINE_boolean(
    "capture_point_log",
    False,
    "Enable capture-point proxy logging diagnostics.",
)
"""
  text, n = re.subn(
      r"\n\ndef get_rl_config\(",
      "\n" + recovery_reward_flags_block + "\n\ndef get_rl_config(",
      text,
      count=1,
  )
  if n == 0:
    text, n2 = re.subn(
        r'(_PUSH_ENTROPY_DELTA\s*=\s*flags\.DEFINE_float\([\s\S]*?\)\n)',
        r"\1" + recovery_reward_flags_block + "\n",
        text,
        count=1,
    )
    if n2 == 0:
      print("[bootstrap] recovery reward flag insertion failed; skipping")
      raise SystemExit(0)

env_overrides_old = """  env_cfg_overrides = {}
  if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
    env_cfg_overrides = json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value)
"""
env_overrides_new = """  env_cfg_overrides = {}
  if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
    env_cfg_overrides = json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value)
  env_cfg_overrides = _merge_overrides(
      env_cfg_overrides, _build_recovery_reward_overrides()
  )
"""
if "_build_recovery_reward_overrides()" in text and env_overrides_old in text:
  text = text.replace(env_overrides_old, env_overrides_new, 1)

text, replaced_helpers = re.subn(
    r"\n# __codex_maxrl_scaffold_v[0-9A-Za-z_]+__\n(?:def _merge_overrides|def _groupwise_binary_weights|def _log_maxrl_scaffold_config)[\s\S]*?\n\ndef main\(argv\):",
    "\n\n" + helpers_block + "\n\ndef main(argv):",
    text,
    count=1,
)

if replaced_helpers == 0:
  text, replaced_helpers = re.subn(
      r"\n\ndef (?:_merge_overrides|_groupwise_binary_weights)\([\s\S]*?\n\ndef main\(argv\):",
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
install_push_line = "  _install_push_mask_patch()\n"
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

if install_push_line not in text:
  if install_line in text:
    text = text.replace(install_line, install_line + install_push_line, 1)
  elif call_line in text:
    text = text.replace(call_line, call_line + install_push_line, 1)
  else:
    print("[bootstrap] push-mask install call anchor missing; skipping")
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
  apply_brax_checkpoint_restore_compat_shim
  apply_g1_recovery_reward_shim
  apply_g1_non_gait_recovery_shim
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
apply_brax_checkpoint_restore_compat_shim
apply_g1_recovery_reward_shim
apply_g1_non_gait_recovery_shim
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
