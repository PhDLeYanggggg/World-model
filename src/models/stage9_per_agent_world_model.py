from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.models.stage9_residual_decoder import predict_linear, sigmoid


class Stage9PerAgentWorldModel:
    def __init__(self, payload: Dict):
        self.payload = payload
        self.residual_clip = float(payload.get("residual_clip", 2.0))

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    def predict_residual(self, dataset: str, horizon: int, x: np.ndarray) -> Dict:
        head = self.payload.get("heads", {}).get(f"{dataset}::{horizon}")
        if not head:
            return {"residual": np.zeros(2, dtype=np.float32), "alpha": 0.0, "failure_probability": 0.0}
        mean = np.asarray(head["x_mean"], dtype=float)
        scale = np.asarray(head["x_scale"], dtype=float)
        rcoef = np.asarray(head["residual_coef"], dtype=float)
        acoef = np.asarray(head["alpha_coef"], dtype=float)
        raw = predict_linear(x, mean, scale, rcoef)
        residual = np.tanh(raw / self.residual_clip) * self.residual_clip
        alpha_logit = predict_linear(x, mean, scale, acoef)
        failure = float(sigmoid(alpha_logit))
        alpha = float(np.clip(float(head.get("alpha_scale", 1.0)) * failure + float(head.get("alpha_bias", 0.0)), 0.0, 1.0))
        return {"residual": residual.astype(np.float32), "alpha": alpha, "failure_probability": failure}
