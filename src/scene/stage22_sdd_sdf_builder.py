from __future__ import annotations

import numpy as np


def boundary_sdf(width: int, height: int, bins: int = 64):
    y, x = np.mgrid[0:bins, 0:bins]
    px = x / max(bins - 1, 1) * width
    py = y / max(bins - 1, 1) * height
    return np.minimum.reduce([px, py, width - px, height - py]).astype("float32")


__all__ = ["boundary_sdf"]

