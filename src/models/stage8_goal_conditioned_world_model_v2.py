from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.models.stage5b6_gated_residual_model import sigmoid
from src.models.stage8_failure_predictor_v2 import Stage8FailurePredictorV2, select_features


class Stage8GoalConditionedWorldModelV2:
    """Deterministic scene/goal/multi-agent bounded residual model.

    Official prediction form:
        strongest_causal_baseline + alpha * bounded_residual

    Top-k goal trajectories are allowed only as diagnostics. This class does
    not implement latent generative modeling or SMC.
    """

    def __init__(self, payload: Dict):
        self.payload = payload
        self.residual_clip = float(payload.get("residual_clip", 3.0))
        fp_path = payload.get("failure_predictor_checkpoint")
        self.failure_predictor = Stage8FailurePredictorV2.load(fp_path) if fp_path and Path(fp_path).exists() else None

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    def predict(self, dataset: str, horizon: int, x: np.ndarray) -> Dict:
        head = self.payload.get("heads", {}).get(f"{dataset}::{horizon}")
        if not head:
            return {"residual": np.zeros(2, dtype=np.float32), "alpha": 0.0, "failure_probability": 0.0}
        mode = head.get("feature_mode", self.payload.get("feature_mode", "scene_goal_multiagent"))
        x_use = select_features(np.asarray(x, dtype=float), mode)
        mean = np.asarray(head["x_mean"], dtype=float)
        scale = np.asarray(head["x_scale"], dtype=float)
        xb = np.concatenate([[1.0], (x_use - mean) / np.maximum(scale, 1e-6)])
        residual_coef = np.asarray(head["residual_coef"], dtype=float)
        raw = xb @ residual_coef
        residual = np.tanh(raw / self.residual_clip) * self.residual_clip
        alpha_coef = np.asarray(head["alpha_coef"], dtype=float)
        learned_alpha = float(sigmoid(xb @ alpha_coef))
        failure_prob = learned_alpha
        if self.failure_predictor is not None:
            failure_prob = self.failure_predictor.predict_proba(dataset, horizon, x)
        alpha = 0.65 * failure_prob + 0.35 * learned_alpha
        alpha = float(np.clip(float(head.get("alpha_scale", 1.0)) * alpha + float(head.get("alpha_bias", 0.0)), 0.0, 1.0))
        if self.payload.get("allow_baseline_fallback", True):
            goal_is_unverified = len(x) >= 23 and x[22] > 0.5
            easy_context = failure_prob < 0.35 and alpha < 0.35
            if goal_is_unverified and easy_context:
                alpha = min(alpha, 0.05)
        return {
            "residual": residual.astype(np.float32),
            "alpha": alpha,
            "failure_probability": float(failure_prob),
            "bounded": True,
            "latent_enabled": False,
            "smc_enabled": False,
        }
