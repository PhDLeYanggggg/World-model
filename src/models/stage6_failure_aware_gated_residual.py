from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.models.baseline_failure_predictor import BaselineFailurePredictor
from src.models.stage5b6_gated_residual_model import sigmoid, trim_features_for_mode


class Stage6FailureAwareGatedResidual:
    def __init__(self, payload: Dict, failure_predictor: BaselineFailurePredictor | None = None):
        self.payload = payload
        self.failure_predictor = failure_predictor
        self.residual_clip = float(payload.get("residual_clip", 4.0))

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fp = None
        fp_path = payload.get("failure_predictor_checkpoint")
        if fp_path and Path(fp_path).exists():
            fp = BaselineFailurePredictor.load(fp_path)
        return cls(payload, fp)

    def predict(self, dataset: str, horizon: int, x: np.ndarray) -> Dict:
        key = f"{dataset}::{horizon}"
        head = self.payload.get("heads", {}).get(key)
        if not head:
            return {"residual": np.zeros(2, dtype=np.float32), "alpha": 0.0, "failure_probability": 0.0}
        feature_mode = head.get("feature_mode", "full")
        x_use = trim_features_for_mode(x, feature_mode)
        mean = np.asarray(head["x_mean"], dtype=np.float64)
        scale = np.asarray(head["x_scale"], dtype=np.float64)
        xb = np.concatenate([[1.0], (x_use - mean) / np.maximum(scale, 1e-6)])
        residual = np.tanh((xb @ np.asarray(head["residual_coef"], dtype=np.float64)) / self.residual_clip) * self.residual_clip
        learned_prob = float(sigmoid(xb @ np.asarray(head["alpha_coef"], dtype=np.float64)))
        failure_prob = self.failure_predictor.predict_proba(x) if self.failure_predictor else learned_prob
        mode = self.payload.get("alpha_mode", "hybrid")
        if mode == "failure_predictor_only":
            alpha = failure_prob
        elif mode == "learned_alpha":
            alpha = learned_prob
        else:
            alpha = 0.7 * failure_prob + 0.3 * learned_prob
        alpha = float(np.clip(head.get("alpha_scale", 1.0) * alpha + head.get("alpha_bias", 0.0), 0.0, 1.0))
        return {"residual": residual.astype(np.float32), "alpha": alpha, "failure_probability": float(failure_prob), "learned_alpha_probability": learned_prob}

