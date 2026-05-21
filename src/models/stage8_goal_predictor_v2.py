from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


class Stage8GoalPredictorV2:
    def __init__(self, payload: Dict):
        self.payload = payload
        self.weights = np.asarray(payload["weights"], dtype=float)

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    def predict(self, record: Dict) -> Dict:
        feats = candidate_features(record)
        if feats.size == 0:
            return {"probabilities": [], "top_goal_indices": [], "entropy": 0.0}
        logits = feats @ self.weights
        probs = softmax(logits)
        order = np.argsort(-probs)
        return {"probabilities": probs.tolist(), "top_goal_indices": [int(i) for i in order[:3]], "entropy": float(-(probs * np.log(np.maximum(probs, 1e-9))).sum())}


def softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / max(float(e.sum()), 1e-9)


def candidate_features(record: Dict) -> np.ndarray:
    dist = np.asarray(record.get("distances_to_goals", []), dtype=float)
    if len(dist) == 0:
        return np.zeros((0, 7), dtype=float)
    angle = np.asarray(record.get("angle_to_goals", [0.0] * len(dist)), dtype=float)
    scale = max(float(np.nanpercentile(dist, 75)), 1.0)
    agent_count = float(record.get("agent_count", 1))
    route = 1.0 if record.get("route_distance_available") else 0.0
    quality = 1.0 if record.get("goal_quality") in {"gold", "silver"} else 0.0
    return np.stack([np.ones_like(dist), -dist / scale, angle, np.arange(len(dist)) / max(len(dist) - 1, 1), np.full_like(dist, min(agent_count / 24.0, 1.0)), np.full_like(dist, route), np.full_like(dist, quality)], axis=1)

