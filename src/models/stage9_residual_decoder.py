from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def fit_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    w = np.sqrt(np.maximum(weights, 1e-6))[:, None]
    return np.linalg.solve((xb * w).T @ (xb * w) + ridge * np.eye(xb.shape[1]), (xb * w).T @ (y * w))


def fit_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray, steps: int = 400, lr: float = 0.06, l2: float = 1e-3) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    y = y.astype(float)
    weights = np.maximum(weights.astype(float), 1e-6)
    coef = np.zeros(xb.shape[1], dtype=float)
    if len(np.unique(y)) < 2:
        prior = np.clip(float(y.mean()) if len(y) else 0.0, 1e-3, 1 - 1e-3)
        coef[0] = np.log(prior / (1 - prior))
        return coef
    for _ in range(steps):
        p = sigmoid(xb @ coef)
        grad = xb.T @ ((p - y) * weights) / max(float(weights.sum()), 1e-6)
        grad[1:] += l2 * coef[1:]
        coef -= lr * grad
    return coef


def predict_linear(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray) -> np.ndarray:
    xb = np.concatenate([[1.0], (x - mean) / np.maximum(scale, 1e-6)])
    return xb @ coef
