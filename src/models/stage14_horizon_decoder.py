from __future__ import annotations

import numpy as np


def bounded_residual_decode(features: np.ndarray, clip: float = 0.1) -> np.ndarray:
    residual = np.zeros((*features.shape[:-1], 2), dtype=np.float32)
    norm = np.linalg.norm(residual, axis=-1, keepdims=True)
    scale = np.minimum(1.0, clip / np.maximum(norm, 1e-6))
    return residual * scale

