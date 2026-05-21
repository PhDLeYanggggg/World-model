from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / max(float(e.sum()), 1e-9)


class GoalIntentPredictor:
    """Causal candidate-goal scorer.

    It scores a scene-level candidate goal dictionary from past trajectory and
    scene features. It never consumes the future endpoint except as a training
    or evaluation label.
    """

    def __init__(self, payload: Dict):
        self.payload = payload
        self.weights = np.asarray(payload["weights"], dtype=float)

    @classmethod
    def load(cls, path: str | Path) -> "GoalIntentPredictor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    def predict(self, record: Dict) -> Dict:
        feats = candidate_feature_matrix(record)
        if feats.size == 0:
            return {"probabilities": [], "top_goal_indices": [], "entropy": 0.0, "confidence": 0.0}
        logits = feats @ self.weights
        probs = softmax(logits)
        order = np.argsort(-probs)
        entropy = float(-(probs * np.log(np.maximum(probs, 1e-9))).sum())
        return {
            "probabilities": probs.tolist(),
            "top_goal_indices": [int(i) for i in order[: min(3, len(order))]],
            "entropy": entropy,
            "confidence": float(probs[order[0]]) if len(order) else 0.0,
        }


def candidate_feature_matrix(record: Dict) -> np.ndarray:
    distances = np.asarray(record.get("distances_to_goals", []), dtype=float)
    if len(distances) == 0:
        return np.zeros((0, 6), dtype=float)
    heading_cos = np.asarray(record.get("heading_cos_to_goals", [0.0] * len(distances)), dtype=float)
    priors = np.asarray(record.get("goal_priors", [1.0 / len(distances)] * len(distances)), dtype=float)
    boundary_distance = float(record.get("boundary_distance", 0.0))
    dist_scale = max(float(np.nanpercentile(np.maximum(distances, 0.0), 75)), 1.0)
    return np.stack(
        [
            np.ones_like(distances),
            -distances / dist_scale,
            heading_cos,
            np.log(np.maximum(priors, 1e-4)),
            np.full_like(distances, min(boundary_distance, 20.0) / 20.0),
            np.arange(len(distances), dtype=float) / max(len(distances) - 1, 1),
        ],
        axis=1,
    )


def train_softmax(records: List[Dict], steps: int = 800, lr: float = 0.05, l2: float = 1e-3) -> np.ndarray:
    w = np.zeros(6, dtype=float)
    usable = [r for r in records if r.get("true_endpoint_cluster_label", -1) >= 0 and r.get("candidate_goal_count", 0) > 1]
    if not usable:
        return w
    for _ in range(steps):
        grad = np.zeros_like(w)
        count = 0
        for r in usable:
            feats = candidate_feature_matrix(r)
            label = int(r["true_endpoint_cluster_label"])
            if label < 0 or label >= len(feats):
                continue
            probs = softmax(feats @ w)
            target = np.zeros(len(probs), dtype=float)
            target[label] = 1.0
            grad += feats.T @ (probs - target)
            count += 1
        if count:
            grad = grad / count + l2 * w
            w -= lr * grad
    return w

