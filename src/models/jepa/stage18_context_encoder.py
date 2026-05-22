from __future__ import annotations

import numpy as np


class Stage18ContextEncoder:
    """Small non-autoregressive encoder for JEPA context features."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    def encode(self, context: np.ndarray) -> np.ndarray:
        x = np.asarray(context, dtype=np.float64)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / np.maximum(self.std, 1e-6)
        return x

