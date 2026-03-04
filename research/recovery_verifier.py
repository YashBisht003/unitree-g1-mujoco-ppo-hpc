"""Binary recovery verifier primitives for push-recovery research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RecoveryThresholds:
  """Thresholds used to classify post-push recovery."""

  min_base_height: float = 0.4
  max_lin_vel_error: float = 0.3
  max_yaw_rate_error: float = 0.75


def compute_binary_conditions(
    base_height: np.ndarray,
    lin_vel_error: np.ndarray,
    yaw_rate_error: np.ndarray,
    done: np.ndarray,
    thresholds: RecoveryThresholds = RecoveryThresholds(),
) -> Mapping[str, np.ndarray]:
  """Returns per-timestep binary conditions in the recovery window."""

  return {
      "upright": (base_height >= thresholds.min_base_height).astype(np.float32),
      "lin_vel_tracking": (
          np.abs(lin_vel_error) <= thresholds.max_lin_vel_error
      ).astype(np.float32),
      "yaw_tracking": (
          np.abs(yaw_rate_error) <= thresholds.max_yaw_rate_error
      ).astype(np.float32),
      "not_terminated": (1.0 - done.astype(np.float32)),
  }


def episode_recovery_success(conditions: Mapping[str, np.ndarray]) -> float:
  """Returns episode-level binary success from timestep condition arrays."""

  if not conditions:
    return 0.0
  stacked = np.stack(list(conditions.values()), axis=0)
  all_ok_per_timestep = np.min(stacked, axis=0)
  return float(np.max(all_ok_per_timestep) > 0.0)
