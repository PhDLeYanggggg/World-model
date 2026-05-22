from __future__ import annotations

import numpy as np


def simple_boundary_sdf(size: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return np.minimum.reduce([xx, yy, size - 1 - xx, size - 1 - yy]).astype("float32") / max(size, 1)

