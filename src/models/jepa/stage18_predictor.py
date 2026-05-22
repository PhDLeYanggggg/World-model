from __future__ import annotations

import numpy as np


class Stage18JEPAPredictor:
    def __init__(self, weights: np.ndarray):
        self.weights = np.asarray(weights, dtype=np.float64)

    def predict(self, context_latent: np.ndarray) -> np.ndarray:
        x = np.asarray(context_latent, dtype=np.float64)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        return x_aug @ self.weights

