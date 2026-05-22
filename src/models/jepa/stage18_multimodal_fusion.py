from __future__ import annotations

import numpy as np


def fuse_modalities(*modalities: np.ndarray) -> np.ndarray:
    arrays = [np.asarray(item, dtype=np.float64) for item in modalities if item is not None]
    if not arrays:
        return np.zeros((1, 1), dtype=np.float64)
    return np.concatenate(arrays, axis=-1)

