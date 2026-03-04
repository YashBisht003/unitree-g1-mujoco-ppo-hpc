"""Scenario grouping utilities for MaxRL-style multi-rollout batches."""

from __future__ import annotations

import numpy as np


def validate_grouping(num_envs: int, group_size: int) -> int:
  """Validates grouping and returns number of scenarios."""

  if group_size <= 0:
    raise ValueError(f"group_size must be > 0, got {group_size}")
  if num_envs % group_size != 0:
    raise ValueError(
        "num_envs must be divisible by group_size: "
        f"num_envs={num_envs}, group_size={group_size}"
    )
  return num_envs // group_size


def scenario_ids(num_envs: int, group_size: int) -> np.ndarray:
  """Returns scenario id for each env index: [0..0,1..1,...]."""

  n_scenarios = validate_grouping(num_envs, group_size)
  return np.repeat(np.arange(n_scenarios, dtype=np.int32), group_size)
