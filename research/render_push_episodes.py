"""Render fixed push-recovery episodes to mp4 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from brax.training import checkpoint as brax_checkpoint
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
import jax
import mediapy as media
import mujoco
import numpy as np
from mujoco_playground import registry


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
  if p.name.isdigit():
    return p
  ckpt_dir = p / "checkpoints"
  if ckpt_dir.is_dir():
    p = ckpt_dir
  step_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.isdigit()]
  if not step_dirs:
    raise ValueError(f"could not find numeric checkpoint step directory under {p}")
  step_dirs.sort(key=lambda d: int(d.name))
  return step_dirs[-1]


def _load_policy_with_fallback(ckpt_step: Path, deterministic: bool):
  load_policy_fn = getattr(ppo_checkpoint, "load_policy", None)
  if callable(load_policy_fn):
    return load_policy_fn(ckpt_step, deterministic=deterministic)

  load_config_fn = getattr(ppo_checkpoint, "load_config", None)
  load_params_fn = getattr(ppo_checkpoint, "load", None)
  if not callable(load_config_fn) or not callable(load_params_fn):
    raise RuntimeError("Unsupported Brax PPO checkpoint API")

  config = load_config_fn(ckpt_step)
  params = load_params_fn(ckpt_step)

  get_network_fn = getattr(ppo_checkpoint, "_get_ppo_network", None)
  if callable(get_network_fn):
    ppo_network = get_network_fn(config, ppo_networks.make_ppo_networks)
  else:
    ppo_network = brax_checkpoint.get_network(config, ppo_networks.make_ppo_networks)
  make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
  return make_inference_fn(params, deterministic=deterministic)


def _build_overrides(
    push_magnitude: float,
    push_angle_deg: float,
    push_interval_s: float,
    command: tuple[float, float, float],
    extra_overrides: dict[str, Any],
) -> dict[str, Any]:
  base = {
      "push_config": {
          "enable": True,
          "magnitude_range": [push_magnitude, push_magnitude],
          "interval_range": [push_interval_s, push_interval_s],
          "direction_mode": "fixed",
          "direction_frame": "body",
          "fixed_angle_deg": push_angle_deg,
          "single_push": True,
      },
      "lin_vel_x": [command[0], command[0]],
      "lin_vel_y": [command[1], command[1]],
      "ang_vel_yaw": [command[2], command[2]],
  }
  return _merge_dict(extra_overrides, base)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--env_name", default="G1JoystickFlatTerrain")
  parser.add_argument("--checkpoint_path", type=Path, required=True)
  parser.add_argument("--output_dir", type=Path, required=True)
  parser.add_argument("--episodes", type=int, default=20)
  parser.add_argument("--episode_length", type=int, default=300)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--push_magnitude", type=float, default=2.5)
  parser.add_argument("--push_angle_deg", type=float, default=0.0)
  parser.add_argument("--push_interval_s", type=float, default=2.0)
  parser.add_argument("--command", default="0.0,0.0,0.0")
  parser.add_argument("--impl", default="jax")
  parser.add_argument("--render_every", type=int, default=2)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--playground_config_overrides", default="")
  args = parser.parse_args()

  parts = [float(x.strip()) for x in args.command.split(",")]
  if len(parts) != 3:
    raise ValueError("--command must have exactly 3 comma-separated floats")
  command = (parts[0], parts[1], parts[2])

  user_overrides = (
      json.loads(args.playground_config_overrides)
      if args.playground_config_overrides
      else {}
  )
  media._config.ffmpeg_name_or_path = imageio_ffmpeg.get_ffmpeg_exe()
  cfg = registry.get_default_config(args.env_name)
  cfg["impl"] = args.impl
  env = registry.load(
      args.env_name,
      config=cfg,
      config_overrides=_build_overrides(
          push_magnitude=args.push_magnitude,
          push_angle_deg=args.push_angle_deg,
          push_interval_s=args.push_interval_s,
          command=command,
          extra_overrides=user_overrides,
      ),
  )

  policy_fn = _load_policy_with_fallback(
      _resolve_checkpoint_step(args.checkpoint_path),
      deterministic=True,
  )

  args.output_dir = args.output_dir.expanduser().resolve()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  metadata = []

  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)

  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  fps = 1.0 / env.dt / max(1, args.render_every)
  rng = jax.random.PRNGKey(args.seed)

  for ep_idx in range(args.episodes):
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    rollout = [state]
    rewards = []
    done_step = None
    push_step = None

    for step_idx in range(args.episode_length):
      act_rng, rng = jax.random.split(rng)
      out = policy_fn(state.obs, act_rng)
      action = out[0] if isinstance(out, tuple) else out
      state = jit_step(state, action)
      rollout.append(state)
      rewards.append(float(np.asarray(state.reward)))

      push = np.asarray(state.info["push"])
      if push_step is None and np.linalg.norm(push) > 1e-6:
        push_step = step_idx + 1
      if bool(np.asarray(state.done)):
        done_step = step_idx + 1
        break

    traj = rollout[:: max(1, args.render_every)]
    frames = env.render(
        traj,
        height=args.height,
        width=args.width,
        scene_option=scene_option,
    )
    video_path = args.output_dir / f"episode_{ep_idx:02d}.mp4"
    media.write_video(video_path.as_posix(), frames, fps=fps)

    metadata.append(
        {
            "episode": ep_idx,
            "video": video_path.name,
            "done_step": done_step,
            "push_step": push_step,
            "total_reward": float(sum(rewards)),
            "success": done_step is None,
        }
    )
    print(
        f"[render] episode={ep_idx:02d} success={done_step is None} "
        f"push_step={push_step} done_step={done_step} video={video_path}"
    )

  metadata_path = args.output_dir / "metadata.json"
  metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  print(f"[render] wrote metadata: {metadata_path}")


if __name__ == "__main__":
  main()
