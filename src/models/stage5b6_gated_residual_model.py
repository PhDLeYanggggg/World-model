from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


@dataclass
class GatedResidualPrediction:
    residual: np.ndarray
    alpha: float
    failure_probability: float


class Stage5B6GatedResidualModel:
    """Baseline-aware deterministic residual model.

    Prediction form:
        strongest_causal_baseline + alpha * bounded_residual

    The JSON payload stores one residual head and one failure-probability head
    per dataset/horizon. No latent generative branch or SMC proposal is used.
    """

    def __init__(self, payload: Dict):
        self.payload = payload
        self.residual_clip = float(payload.get("residual_clip", 4.0))

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    def predict(self, dataset: str, horizon: int, x: np.ndarray, use_interaction: bool = True) -> GatedResidualPrediction:
        key = f"{dataset}::{horizon}"
        head = self.payload.get("heads", {}).get(key)
        if not head:
            return GatedResidualPrediction(np.zeros(2, dtype=np.float32), 0.0, 0.0)
        feature_mode = head.get("feature_mode", "full")
        x_use = trim_features_for_mode(x, feature_mode if use_interaction else "no_interaction")
        mean = np.asarray(head["x_mean"], dtype=np.float64)
        scale = np.asarray(head["x_scale"], dtype=np.float64)
        xz = (x_use - mean) / np.maximum(scale, 1e-6)
        xb = np.concatenate([np.asarray([1.0]), xz])
        residual_coef = np.asarray(head["residual_coef"], dtype=np.float64)
        raw_residual = xb @ residual_coef
        residual = np.tanh(raw_residual / self.residual_clip) * self.residual_clip
        failure_w = np.asarray(head["failure_coef"], dtype=np.float64)
        failure_prob = float(sigmoid(xb @ failure_w))
        alpha_scale = float(head.get("alpha_scale", 1.0))
        alpha_bias = float(head.get("alpha_bias", 0.0))
        alpha = float(np.clip(alpha_scale * failure_prob + alpha_bias, 0.0, 1.0))
        return GatedResidualPrediction(residual.astype(np.float32), alpha, failure_prob)


def trim_features_for_mode(x: np.ndarray, feature_mode: str) -> np.ndarray:
    """Feature blocks are [base 18 | interaction 9]."""
    x = np.asarray(x, dtype=np.float64)
    if feature_mode == "no_interaction":
        return x[:18]
    if feature_mode == "scalar_interaction":
        return x[:22]
    return x


def fit_weighted_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    w = np.sqrt(np.maximum(weights, 1e-6))[:, None]
    xb = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    xw = xb * w
    yw = y * w
    return np.linalg.solve(xw.T @ xw + ridge * np.eye(xb.shape[1]), xw.T @ yw)


def fit_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray, steps: int = 300, lr: float = 0.05, l2: float = 1e-3) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    y = y.astype(np.float64)
    weights = np.maximum(weights.astype(np.float64), 1e-6)
    w = np.zeros(xb.shape[1], dtype=np.float64)
    if len(np.unique(y)) < 2:
        prior = np.clip(y.mean() if len(y) else 0.0, 1e-3, 1.0 - 1e-3)
        w[0] = np.log(prior / (1.0 - prior))
        return w
    for _ in range(steps):
        p = sigmoid(xb @ w)
        grad = (xb.T @ ((p - y) * weights)) / max(float(weights.sum()), 1e-6)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w

