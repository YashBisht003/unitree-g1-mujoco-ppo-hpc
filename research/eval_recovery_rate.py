"""Evaluate push-recovery rate versus push magnitude for a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Callable

from brax.training import checkpoint as brax_checkpoint
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
import jax
import numpy as np
from mujoco_playground import registry


def _parse_magnitudes(text: str) -> list[float]:
  text = text.strip()
  if not text:
    return [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]
  if text.startswith("["):
    values = json.loads(text)
    return [float(v) for v in values]
  return [float(v.strip()) for v in text.split(",") if v.strip()]


def _parse_command(text: str | None) -> tuple[float, float, float] | None:
  if not text:
    return None
  parts = [p.strip() for p in text.split(",") if p.strip()]
  if len(parts) != 3:
    raise ValueError(
        "--command must be a comma-separated triple: lin_x,lin_y,yaw"
    )
  return (float(parts[0]), float(parts[1]), float(parts[2]))


def _merge_dict(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
  out = dict(dst)
  for k, v in src.items():
    if isinstance(v, dict) and isinstance(out.get(k), dict):
      out[k] = _merge_dict(out[k], v)
    else:
      out[k] = v
  return out


def _resolve_checkpoint_step(path: Path) -> Path:
  p = path.expanduser().resolve()
  if not p.exists():
    raise FileNotFoundError(f"checkpoint path does not exist: {p}")
  if not p.is_dir():
    raise ValueError(f"checkpoint path must be a directory: {p}")

  if p.name.isdigit():
    return p

  ckpt_dir = p / "checkpoints"
  if ckpt_dir.is_dir():
    p = ckpt_dir

  step_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.isdigit()]
  if not step_dirs:
    raise ValueError(
      "could not find numeric checkpoint step directory under "
      f"{path.expanduser().resolve()}"
    )
  step_dirs.sort(key=lambda d: int(d.name))
  return step_dirs[-1]


def _load_json_if_exists(path: Path) -> dict[str, Any]:
  if path.is_file():
    return json.loads(path.read_text(encoding="utf-8"))
  return {}


def _load_policy_with_fallback(
    ckpt_step: Path, deterministic: bool
) -> Callable[[Any, Any], Any]:
  """Loads PPO policy inference function across Brax API variants."""

  load_policy_fn = getattr(ppo_checkpoint, "load_policy", None)
  if callable(load_policy_fn):
    return load_policy_fn(ckpt_step, deterministic=deterministic)

  load_config_fn = getattr(ppo_checkpoint, "load_config", None)
  load_params_fn = getattr(ppo_checkpoint, "load", None)
  if not callable(load_config_fn) or not callable(load_params_fn):
    raise RuntimeError(
        "Unsupported Brax PPO checkpoint API: missing load_policy and "
        "load/load_config fallback functions."
    )

  config = load_config_fn(ckpt_step)
  params = load_params_fn(ckpt_step)

  get_network_fn = getattr(ppo_checkpoint, "_get_ppo_network", None)
  if callable(get_network_fn):
    ppo_network = get_network_fn(config, ppo_networks.make_ppo_networks)
  else:
    ppo_network = brax_checkpoint.get_network(config, ppo_networks.make_ppo_networks)

  make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
  return make_inference_fn(params, deterministic=deterministic)


def _build_config_overrides(
    user_overrides: dict[str, Any],
    push_magnitude: float,
    push_interval_s: float,
    command: tuple[float, float, float] | None,
) -> dict[str, Any]:
  override = {
      "push_config": {
          "enable": True,
          "magnitude_range": [push_magnitude, push_magnitude],
          "interval_range": [push_interval_s, push_interval_s],
      }
  }
  if command is not None:
    override.update(
        {
            "lin_vel_x": [command[0], command[0]],
            "lin_vel_y": [command[1], command[1]],
            "ang_vel_yaw": [command[2], command[2]],
        }
    )
  return _merge_dict(user_overrides, override)


def _evaluate_one_magnitude(
    env_name: str,
    impl: str,
    policy_fn,
    episodes: int,
    batch_size: int,
    episode_length: int,
    recovery_window_s: float,
    seed: int,
    config_overrides: dict[str, Any],
    include_no_push_as_fail: bool,
) -> dict[str, Any]:
  env_cfg = registry.get_default_config(env_name)
  env_cfg["impl"] = impl
  env = registry.load(
      env_name,
      config=env_cfg,
      config_overrides=config_overrides,
  )

  recovery_window_steps = max(1, int(math.ceil(recovery_window_s / float(env.dt))))

  v_reset = jax.jit(jax.vmap(env.reset))
  v_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0)))
  v_split = jax.jit(jax.vmap(lambda k: jax.random.split(k, 2)))

  def _act(obs, key):
    out = policy_fn(obs, key)
    return out[0] if isinstance(out, tuple) else out

  v_act = jax.jit(jax.vmap(_act))

  total_success = 0
  total_no_push = 0
  total_eval = 0
  total_episode_reward_all = 0.0
  total_episode_reward_eval = 0.0
  total_post_push_reward = 0.0
  total_survival_steps_after_push = 0
  remaining = episodes
  rng = jax.random.PRNGKey(seed)

  while remaining > 0:
    n = min(batch_size, remaining)
    remaining -= n

    rng, reset_key = jax.random.split(rng)
    keys = jax.random.split(reset_key, n)
    state = v_reset(keys)
    act_keys = keys

    pushed = np.zeros((n,), dtype=bool)
    failed_after_push = np.zeros((n,), dtype=bool)
    window_left = np.zeros((n,), dtype=np.int32)
    done_seen = np.zeros((n,), dtype=bool)
    episode_reward = np.zeros((n,), dtype=np.float64)
    post_push_reward = np.zeros((n,), dtype=np.float64)
    survival_steps_after_push = np.zeros((n,), dtype=np.int32)

    for _ in range(episode_length):
      split_keys = v_split(act_keys)
      act_keys = split_keys[:, 0, :]
      sample_keys = split_keys[:, 1, :]

      action = v_act(state.obs, sample_keys)
      state = v_step(state, action)

      reward = np.asarray(state.reward, dtype=np.float64)
      if reward.ndim == 0:
        reward = np.full((n,), float(reward), dtype=np.float64)
      else:
        reward = reward.reshape(n)
      done = np.asarray(state.done) > 0.5
      active = ~done_seen
      episode_reward[active] += reward[active]
      push_vec = state.info.get("push", None) if hasattr(state.info, "get") else None
      if push_vec is None:
        # Some env variants do not expose push vectors in state.info.
        # Fallback: treat this step as "post-push" so the evaluator remains usable.
        push_now = np.ones((n,), dtype=bool)
      else:
        push = np.asarray(push_vec)
        if push.ndim == 1:
          if push.shape[0] == n:
            push_now = np.abs(push) > 1e-6
          else:
            push_now = np.full((n,), np.linalg.norm(push) > 1e-6, dtype=bool)
        else:
          push_now = np.linalg.norm(push, axis=-1) > 1e-6
      new_push = (~pushed) & push_now

      pushed[new_push] = True
      window_left[new_push] = recovery_window_steps

      active_window = window_left > 0
      failed_after_push[active_window & done] = True
      post_push_reward[active_window & active] += reward[active_window & active]
      survival_steps_after_push[active_window & active & (~done)] += 1
      window_left[active_window] -= 1
      done_seen |= done

      if np.all(pushed) and np.all(window_left <= 0):
        break

    no_push = int((~pushed).sum())
    success = pushed & (~failed_after_push)
    eval_mask = np.ones((n,), dtype=bool) if include_no_push_as_fail else pushed

    total_no_push += no_push
    total_episode_reward_all += float(episode_reward.sum())
    total_episode_reward_eval += float(episode_reward[eval_mask].sum())
    total_post_push_reward += float(post_push_reward[pushed].sum())
    total_survival_steps_after_push += int(survival_steps_after_push[pushed].sum())
    if include_no_push_as_fail:
      total_eval += n
      total_success += int(success.sum())
    else:
      valid = int(pushed.sum())
      total_eval += valid
      total_success += int(success.sum())

  rate = float(total_success / total_eval) if total_eval > 0 else 0.0
  return {
      "episodes_requested": episodes,
      "episodes_evaluated": total_eval,
      "successes": total_success,
      "falls_after_push": int(total_eval - total_success),
      "recovery_rate": rate,
      "no_push_episodes": total_no_push,
      "recovery_window_s": recovery_window_s,
      "recovery_window_steps": recovery_window_steps,
      "episode_length": episode_length,
      "mean_episode_reward_all": (
          float(total_episode_reward_all / episodes) if episodes > 0 else 0.0
      ),
      "mean_episode_reward_eval": (
          float(total_episode_reward_eval / total_eval) if total_eval > 0 else 0.0
      ),
      "mean_post_push_reward": (
          float(total_post_push_reward / total_eval) if total_eval > 0 else 0.0
      ),
      "mean_survival_steps_after_push": (
          float(total_survival_steps_after_push / total_eval)
          if total_eval > 0
          else 0.0
      ),
      "mean_survival_time_s_after_push": (
          float(total_survival_steps_after_push / total_eval) * float(env.dt)
          if total_eval > 0
          else 0.0
      ),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint_path", type=Path, required=True)
  parser.add_argument("--env_name", type=str, default="G1JoystickRoughTerrain")
  parser.add_argument("--impl", type=str, default="jax")
  parser.add_argument("--episodes_per_magnitude", type=int, default=200)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--episode_length", type=int, default=1000)
  parser.add_argument("--recovery_window_s", type=float, default=5.0)
  parser.add_argument("--push_interval_s", type=float, default=2.0)
  parser.add_argument("--push_magnitudes", type=str, default="2,4,6,8,10,12,15")
  parser.add_argument(
      "--command",
      type=str,
      default="",
      help="Optional fixed command lin_x,lin_y,yaw (example: 0.8,0.0,0.0)",
  )
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--deterministic", action="store_true")
  parser.add_argument(
      "--include_no_push_as_fail",
      action="store_true",
      default=False,
      help="Treat episodes with no push event as failures (default: exclude).",
  )
  parser.add_argument("--playground_config_overrides", type=str, default="")
  parser.add_argument(
      "--output_json",
      type=Path,
      default=None,
      help="Path to save JSON results (default: research/results/<timestamp>.json)",
  )
  parser.add_argument(
      "--output_csv",
      type=Path,
      default=None,
      help="Optional CSV output path (default: alongside JSON with .csv suffix).",
  )
  args = parser.parse_args()

  if args.episodes_per_magnitude <= 0:
    raise ValueError("--episodes_per_magnitude must be > 0")
  if args.batch_size <= 0:
    raise ValueError("--batch_size must be > 0")
  if args.episode_length <= 0:
    raise ValueError("--episode_length must be > 0")
  if args.recovery_window_s <= 0:
    raise ValueError("--recovery_window_s must be > 0")
  if args.push_interval_s <= 0:
    raise ValueError("--push_interval_s must be > 0")

  magnitudes = _parse_magnitudes(args.push_magnitudes)
  cmd = _parse_command(args.command if args.command else None)
  user_overrides = (
      json.loads(args.playground_config_overrides)
      if args.playground_config_overrides
      else {}
  )

  ckpt_step = _resolve_checkpoint_step(args.checkpoint_path)
  policy_fn = _load_policy_with_fallback(
      ckpt_step,
      deterministic=args.deterministic,
  )

  timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  if args.output_json is None:
    output_json = Path("research") / "results" / f"recovery_rate_{timestamp}.json"
  else:
    output_json = args.output_json
  output_json = output_json.expanduser().resolve()
  output_json.parent.mkdir(parents=True, exist_ok=True)

  if args.output_csv is None:
    output_csv = output_json.with_suffix(".csv")
  else:
    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

  all_results: list[dict[str, Any]] = []
  for i, mag in enumerate(magnitudes):
    overrides = _build_config_overrides(
        user_overrides=user_overrides,
        push_magnitude=float(mag),
        push_interval_s=args.push_interval_s,
        command=cmd,
    )
    result = _evaluate_one_magnitude(
        env_name=args.env_name,
        impl=args.impl,
        policy_fn=policy_fn,
        episodes=args.episodes_per_magnitude,
        batch_size=args.batch_size,
        episode_length=args.episode_length,
        recovery_window_s=args.recovery_window_s,
        seed=args.seed + i,
        config_overrides=overrides,
        include_no_push_as_fail=args.include_no_push_as_fail,
    )
    result["push_magnitude"] = float(mag)
    all_results.append(result)
    print(
        "[eval] magnitude="
        f"{mag:.3f} recovery_rate={result['recovery_rate']:.4f} "
        f"successes={result['successes']}/{result['episodes_evaluated']} "
        f"mean_reward={result['mean_episode_reward_eval']:.3f} "
        f"mean_post_push_reward={result['mean_post_push_reward']:.3f} "
        f"no_push={result['no_push_episodes']}"
    )

  payload = {
      "timestamp": timestamp,
      "checkpoint_step_dir": str(ckpt_step),
      "env_name": args.env_name,
      "impl": args.impl,
      "episodes_per_magnitude": args.episodes_per_magnitude,
      "batch_size": args.batch_size,
      "episode_length": args.episode_length,
      "recovery_window_s": args.recovery_window_s,
      "push_interval_s": args.push_interval_s,
      "push_magnitudes": magnitudes,
      "include_no_push_as_fail": args.include_no_push_as_fail,
      "command": list(cmd) if cmd is not None else None,
      "playground_config_overrides": user_overrides,
      "results": all_results,
  }
  output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

  with output_csv.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=[
            "push_magnitude",
            "recovery_rate",
            "successes",
            "falls_after_push",
            "episodes_evaluated",
            "episodes_requested",
            "no_push_episodes",
            "recovery_window_s",
            "recovery_window_steps",
            "episode_length",
            "mean_episode_reward_all",
            "mean_episode_reward_eval",
            "mean_post_push_reward",
            "mean_survival_steps_after_push",
            "mean_survival_time_s_after_push",
        ],
    )
    writer.writeheader()
    for row in all_results:
      writer.writerow({k: row[k] for k in writer.fieldnames})

  print(f"[eval] wrote JSON: {output_json}")
  print(f"[eval] wrote CSV : {output_csv}")


if __name__ == "__main__":
  main()
