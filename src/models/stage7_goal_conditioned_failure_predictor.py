from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.models.stage5b6_gated_residual_model import sigmoid


class Stage7GoalConditionedFailurePredictor:
    def __init__(self, payload: Dict):
        self.payload = payload

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    def predict_proba(self, dataset: str, horizon: int, x: np.ndarray) -> float:
        head = self.payload.get("heads", {}).get(f"{dataset}::{horizon}") or self.payload.get("global_head")
        if not head:
            return 0.0
        mean = np.asarray(head["x_mean"], dtype=float)
        scale = np.asarray(head["x_scale"], dtype=float)
        coef = np.asarray(head["coef"], dtype=float)
        xb = np.concatenate([[1.0], (np.asarray(x, dtype=float) - mean) / np.maximum(scale, 1e-6)])
        return float(sigmoid(xb @ coef))

    def predict_reason(self, x: np.ndarray) -> str:
        if len(x) >= 8 and x[-6] > 0.65:
            return "wrong_goal_or_goal_ambiguity"
        if len(x) >= 27 and x[21] > 0.5:
            return "interaction"
        if len(x) >= 11 and x[10] > 0.4:
            return "turn"
        return "long_horizon_drift"

