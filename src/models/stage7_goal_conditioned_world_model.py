from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.models.stage5b6_gated_residual_model import sigmoid
from src.models.stage7_goal_conditioned_failure_predictor import Stage7GoalConditionedFailurePredictor


class Stage7GoalConditionedWorldModel:
    """Deterministic scene/goal-grounded bounded residual model.

    prediction = strongest_causal_baseline + alpha * bounded_residual(goal, scene, interaction)
    """

    def __init__(self, payload: Dict):
        self.payload = payload
        self.residual_clip = float(payload.get("residual_clip", 4.0))
        fp_path = payload.get("failure_predictor_checkpoint")
        self.failure_predictor = Stage7GoalConditionedFailurePredictor.load(fp_path) if fp_path and Path(fp_path).exists() else None

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    def predict(self, dataset: str, horizon: int, x: np.ndarray) -> Dict:
        head = self.payload.get("heads", {}).get(f"{dataset}::{horizon}")
        if not head:
            return {"residual": np.zeros(2, dtype=np.float32), "alpha": 0.0, "failure_probability": 0.0, "goal_confidence": 0.0}
        mean = np.asarray(head["x_mean"], dtype=float)
        scale = np.asarray(head["x_scale"], dtype=float)
        coef = np.asarray(head["residual_coef"], dtype=float)
        alpha_coef = np.asarray(head["alpha_coef"], dtype=float)
        xb = np.concatenate([[1.0], (np.asarray(x, dtype=float) - mean) / np.maximum(scale, 1e-6)])
        residual = np.tanh((xb @ coef) / self.residual_clip) * self.residual_clip
        learned_alpha = float(sigmoid(xb @ alpha_coef))
        fp = learned_alpha
        if self.failure_predictor:
            fp_head = self.failure_predictor.payload.get("heads", {}).get(f"{dataset}::{horizon}") or self.failure_predictor.payload.get("global_head")
            expected = len(fp_head.get("x_mean", [])) if fp_head else len(x)
            if expected == len(x):
                fp = self.failure_predictor.predict_proba(dataset, horizon, x)
        mode = self.payload.get("alpha_mode", "goal_conditioned_failure")
        if mode == "learned_alpha_only":
            alpha = learned_alpha
        elif mode == "failure_predictor_only":
            alpha = fp
        else:
            alpha = 0.75 * fp + 0.25 * learned_alpha
        alpha = float(np.clip(float(head.get("alpha_scale", 1.0)) * alpha + float(head.get("alpha_bias", 0.0)), 0.0, 1.0))
        goal_conf = float(x[-8]) if len(x) >= 8 else 0.0
        if self.payload.get("allow_baseline_fallback", True) and goal_conf < 0.15 and fp < 0.35:
            alpha = min(alpha, 0.05)
        return {"residual": residual.astype(np.float32), "alpha": alpha, "failure_probability": float(fp), "goal_confidence": goal_conf}
