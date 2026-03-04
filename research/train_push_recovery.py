"""Research launcher scaffold for push-recovery ablations."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _map_mode(mode: str) -> str:
  if mode == "ppo_dense":
    return "ppo"
  if mode == "maxrl_binary":
    return "maxrl_binary"
  if mode == "maxrl_t":
    return "maxrl_temporal"
  raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo_root", type=Path, required=True)
  parser.add_argument("--checkpoint_path", type=Path, required=True)
  parser.add_argument("--mode", choices=["ppo_dense", "maxrl_binary", "maxrl_t"], default="ppo_dense")
  parser.add_argument("--num_timesteps", type=int, default=50_000_000)
  parser.add_argument("--num_envs", type=int, default=1024)
  parser.add_argument("--num_eval_envs", type=int, default=128)
  parser.add_argument("--scenario_group_size", type=int, default=8)
  parser.add_argument("--suffix", type=str, default="g1-push-research")
  parser.add_argument("--playground_config_overrides", type=str, default="")
  parser.add_argument("--dry_run", action="store_true")
  args = parser.parse_args()

  train_py = args.repo_root / "mujoco_playground" / "learning" / "train_jax_ppo.py"
  if not train_py.exists():
    raise FileNotFoundError(f"Missing {train_py}")

  cmd = [
      "python",
      str(train_py),
      "--env_name=G1JoystickRoughTerrain",
      "--domain_randomization=True",
      f"--num_timesteps={args.num_timesteps}",
      f"--num_envs={args.num_envs}",
      f"--num_eval_envs={args.num_eval_envs}",
      "--policy_hidden_layer_sizes=512,256,128",
      "--value_hidden_layer_sizes=512,256,128",
      "--policy_obs_key=state",
      "--value_obs_key=privileged_state",
      "--seed=1",
      f"--suffix={args.suffix}",
      f"--adv_mode={_map_mode(args.mode)}",
      f"--scenario_group_size={args.scenario_group_size}",
      "--maxrl_log_only=False",
      "--maxrl_scenario_key=push_cfg",
      "--maxrl_verbose=True",
      f"--load_checkpoint_path={args.checkpoint_path}",
      "--use_tb=True",
      "--use_wandb=False",
  ]
  if args.playground_config_overrides:
    json.loads(args.playground_config_overrides)
    cmd.append(f"--playground_config_overrides={args.playground_config_overrides}")

  print(" ".join(cmd))
  if args.dry_run:
    return
  subprocess.run(cmd, check=True, cwd=str(args.repo_root / "mujoco_playground"))


if __name__ == "__main__":
  main()
