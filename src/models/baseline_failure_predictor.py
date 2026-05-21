from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.models.stage5b6_gated_residual_model import sigmoid


class BaselineFailurePredictor:
    def __init__(self, payload: Dict):
        self.payload = payload

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    def predict_proba(self, x: np.ndarray) -> float:
        mean = np.asarray(self.payload["x_mean"], dtype=np.float64)
        scale = np.asarray(self.payload["x_scale"], dtype=np.float64)
        coef = np.asarray(self.payload["coef"], dtype=np.float64)
        xz = (np.asarray(x, dtype=np.float64) - mean) / np.maximum(scale, 1e-6)
        xb = np.concatenate([[1.0], xz])
        return float(sigmoid(xb @ coef))

