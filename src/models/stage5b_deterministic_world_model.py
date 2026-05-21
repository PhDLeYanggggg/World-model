from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.evaluation.baseline_benchmark_stage5b import rollout


@dataclass
class LinearResidualHead:
    coef_x: List[float]
    coef_y: List[float]
    baseline: str
    mode: str

    def predict(self, features: np.ndarray) -> np.ndarray:
        coef_x = np.asarray(self.coef_x, dtype=np.float32)
        coef_y = np.asarray(self.coef_y, dtype=np.float32)
        return np.stack([features @ coef_x, features @ coef_y], axis=-1)


class Stage5BDeterministicWorldModel:
    """A deliberately small deterministic residual model for Stage 5B gates.

    This is not the final foundation model. It learns causal residual corrections over
    each dataset's strongest causal kinematic prior and exists to test whether learned
    dynamics can beat the repaired baselines before enabling latent models or SMC.
    """

    def __init__(self, heads: Dict[str, LinearResidualHead]):
        self.heads = heads

    def predict_rollout(self, dataset: str, history: np.ndarray, horizon: int, dt: float) -> np.ndarray:
        head = self.heads[dataset]
        pred = rollout(history, horizon, dt, head.baseline)[1 : horizon + 1]
        for t in range(horizon):
            features = make_features(history, step=t + 1, horizon=horizon)
            residual = head.predict(features)
            pred[t, :, 0:2] += residual
            if t > 0:
                pred[t, :, 2:4] = (pred[t, :, 0:2] - pred[t - 1, :, 0:2]) / max(dt, 1e-6)
            pred[t, :, 6] = np.arctan2(pred[t, :, 3], pred[t, :, 2])
            pred[t, :, 7] = np.linalg.norm(pred[t, :, 2:4], axis=1)
        return pred

    def save(self, path: str | Path) -> None:
        payload = {"heads": {k: v.__dict__ for k, v in self.heads.items()}}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Stage5BDeterministicWorldModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({k: LinearResidualHead(**v) for k, v in payload["heads"].items()})


def make_features(history: np.ndarray, step: int, horizon: int) -> np.ndarray:
    last = history[-1]
    prev = history[-2] if history.shape[0] >= 2 else last
    dtheta = np.angle(np.exp(1j * (last[:, 6] - prev[:, 6])))
    step_frac = float(step) / max(float(horizon), 1.0)
    ones = np.ones((history.shape[1], 1), dtype=np.float32)
    return np.concatenate(
        [
            ones,
            last[:, 2:4],
            last[:, 4:6],
            last[:, 7:8],
            np.sin(last[:, 6:7]),
            np.cos(last[:, 6:7]),
            dtheta[:, None],
            np.full((history.shape[1], 1), step_frac, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
