from __future__ import annotations

from typing import Dict, List

import numpy as np


def rectangular_boundary(points: np.ndarray, margin_ratio: float = 0.08) -> List[List[float]]:
    """Build a conservative scene boundary from observed coordinates.

    This is an inferred boundary, not a true annotated walkable region.
    """
    if points.size == 0:
        return [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    lo = np.nanmin(points, axis=0)
    hi = np.nanmax(points, axis=0)
    span = np.maximum(hi - lo, 1.0)
    margin = span * margin_ratio
    x0, y0 = lo - margin
    x1, y1 = hi + margin
    return [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]]


def boundary_summary(polygon: List[List[float]]) -> Dict:
    arr = np.asarray(polygon, dtype=float)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    return {
        "min_x": float(lo[0]),
        "min_y": float(lo[1]),
        "max_x": float(hi[0]),
        "max_y": float(hi[1]),
        "area": float(max(hi[0] - lo[0], 0.0) * max(hi[1] - lo[1], 0.0)),
    }

