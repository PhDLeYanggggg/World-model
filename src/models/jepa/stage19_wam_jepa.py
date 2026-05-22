from __future__ import annotations

import numpy as np


class Stage19WAMJEPA:
    """Non-generative WAM-style JEPA representation model."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float64)

    def encode(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x @ self.weights[:-1] + self.weights[-1]

