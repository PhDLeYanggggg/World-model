from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.models.stage5b6_gated_residual_model import sigmoid


class Stage8FailurePredictorV2:
    """Scene/goal/multi-agent baseline-failure predictor.

    This is a causal diagnostic model. It predicts whether the strongest
    causal baseline is likely to fail from past states, goal candidates, and
    multi-agent context. It never consumes future endpoints at inference time.
    """

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
        x_use = select_features(np.asarray(x, dtype=float), head.get("feature_mode", "scene_goal_multiagent"))
        mean = np.asarray(head["x_mean"], dtype=float)
        scale = np.asarray(head["x_scale"], dtype=float)
        coef = np.asarray(head["coef"], dtype=float)
        xb = np.concatenate([[1.0], (x_use - mean) / np.maximum(scale, 1e-6)])
        return float(sigmoid(xb @ coef))


def select_features(x: np.ndarray, mode: str) -> np.ndarray:
    """Stage 8 feature vector is [base15 | goal8].

    base15 includes kinematics, scalar multi-agent density/proximity, horizon,
    coordinate, and traffic-domain indicators. goal8 includes scene/goal
    candidate features and annotation quality flags.
    """

    if mode == "no_scene_goal":
        return x[:15]
    if mode == "scene_only":
        return np.concatenate([x[:15], x[[20, 21, 22]]])
    if mode == "goal_only":
        return np.concatenate([x[:13], x[15:21]])
    if mode == "scene_goal":
        return np.concatenate([x[:15], x[15:]])
    return x
