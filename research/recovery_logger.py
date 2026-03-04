"""Lightweight logging helpers for recovery-signature experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RecoveryRecord:
  run_name: str
  scenario_id: int
  rollout_id: int
  success: float
  signature_density_mean: float


def signature_density(conditions: Mapping[str, np.ndarray]) -> np.ndarray:
  """Returns the per-timestep fraction of satisfied binary conditions."""

  if not conditions:
    return np.zeros((0,), dtype=np.float32)
  stacked = np.stack(list(conditions.values()), axis=0).astype(np.float32)
  return np.mean(stacked, axis=0)


def append_record(path: str | Path, record: RecoveryRecord) -> None:
  """Appends a JSONL record."""

  p = Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  with p.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(asdict(record)) + "\n")
