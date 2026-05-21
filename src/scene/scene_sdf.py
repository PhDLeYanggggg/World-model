from __future__ import annotations

from typing import Dict, List

import numpy as np


def boundary_distance(point: np.ndarray, boundary: Dict) -> float:
    x, y = float(point[0]), float(point[1])
    return float(min(x - boundary["min_x"], boundary["max_x"] - x, y - boundary["min_y"], boundary["max_y"] - y))


def goal_features(point: np.ndarray, velocity: np.ndarray, goals: List[Dict]) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    speed = float(np.linalg.norm(velocity))
    feats = []
    for goal in goals:
        center = np.asarray(goal["center"], dtype=float)
        vec = center - point
        dist = float(np.linalg.norm(vec))
        direction = vec / max(dist, 1e-6)
        heading_cos = float(np.dot(direction, velocity / max(speed, 1e-6))) if speed > 1e-6 else 0.0
        feats.append([dist, heading_cos, float(goal.get("support_fraction", 0.0)), float(goal.get("radius", 0.0))])
    return np.asarray(feats, dtype=float)


def walkability_sdf_sample(point: np.ndarray, boundary: Dict) -> Dict:
    dist = boundary_distance(point, boundary)
    return {
        "boundary_distance": dist,
        "inside_inferred_walkable_area": bool(dist >= 0.0),
        "walkability_source": "inferred_bbox_from_observed_trajectories",
    }

