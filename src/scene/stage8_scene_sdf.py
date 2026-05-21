from __future__ import annotations

from typing import Dict

import numpy as np


def point_to_boundary_sdf(point: np.ndarray, boundary_summary: Dict) -> float:
    x, y = float(point[0]), float(point[1])
    return float(min(x - boundary_summary["min_x"], boundary_summary["max_x"] - x, y - boundary_summary["min_y"], boundary_summary["max_y"] - y))


def point_to_goal_distances(point: np.ndarray, goals: list[Dict]) -> list[float]:
    p = np.asarray(point, dtype=float)
    return [float(np.linalg.norm(p - np.asarray(goal["center"], dtype=float))) for goal in goals]

