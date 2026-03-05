"""Core MaxRL estimator helpers for binary verifier outcomes.

This is a NumPy reference implementation for offline analysis utilities.
Active training-time weighting is injected via bootstrap shim into
`learning/train_jax_ppo.py`.
"""

from __future__ import annotations

import numpy as np


def maxrl_weights(successes: np.ndarray) -> np.ndarray | None:
  """Returns per-rollout MaxRL weights (r_i / K) or None when K == 0."""

  s = np.asarray(successes, dtype=np.float32).reshape(-1)
  k = float(np.sum(s))
  if k <= 0.0:
    return None
  return s / k


def apply_maxrl_weights(advantages: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
  """Applies rollout-level MaxRL weights to rollout advantages."""

  if weights is None:
    return np.zeros_like(advantages)
  if advantages.shape[0] != weights.shape[0]:
    raise ValueError(
        "shape mismatch: advantages batch dimension must match weights: "
        f"{advantages.shape[0]} != {weights.shape[0]}"
    )
  return advantages * weights[:, None]
