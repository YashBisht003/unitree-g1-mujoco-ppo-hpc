"""Temporal MaxRL extension helpers (research scaffold)."""

from __future__ import annotations

import numpy as np


def temporal_maxrl_weights(signature_density: np.ndarray) -> np.ndarray | None:
  """Returns per-rollout soft MaxRL weights from signature density.

  Args:
    signature_density: shape [N], values in [0, 1] for one timestep.
  """

  s = np.asarray(signature_density, dtype=np.float32).reshape(-1)
  mass = float(np.sum(s))
  if mass <= 1e-8:
    return None
  return s / mass


def apply_temporal_weights(advantages_t: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
  """Applies per-rollout temporal weights at one timestep."""

  if weights is None:
    return np.zeros_like(advantages_t)
  if advantages_t.shape[0] != weights.shape[0]:
    raise ValueError(
        "shape mismatch: advantages_t batch dimension must match weights: "
        f"{advantages_t.shape[0]} != {weights.shape[0]}"
    )
  return advantages_t * weights
