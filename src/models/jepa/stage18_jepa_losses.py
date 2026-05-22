from __future__ import annotations

import numpy as np


def jepa_latent_l2_loss(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((np.asarray(predicted) - np.asarray(target)) ** 2))


def variance_regularization(latent: np.ndarray) -> float:
    variance = np.var(np.asarray(latent), axis=0)
    return float(np.mean(np.maximum(0.0, 1e-4 - variance)))

