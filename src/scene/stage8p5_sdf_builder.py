from __future__ import annotations

import numpy as np


def polygon_distance(point, polygon) -> float:
    if not polygon:
        return 0.0
    p = np.asarray(point, dtype=float)
    pts = np.asarray(polygon, dtype=float)
    d = np.linalg.norm(pts - p[None, :], axis=1)
    return float(d.min()) if len(d) else 0.0


def goal_distance_fields(goal_regions):
    return [{"goal_id": g.get("goal_id"), "center": g.get("center"), "radius": g.get("radius", 1.0), "field": "analytic_euclidean"} for g in goal_regions]
