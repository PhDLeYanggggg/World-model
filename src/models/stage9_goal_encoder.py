from __future__ import annotations

import numpy as np


def encode_goal(pack: dict, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    goals = pack.get("goal_regions", [])
    if not goals:
        return np.zeros(8, dtype=float)
    centers = np.asarray([g.get("center", [0.0, 0.0]) for g in goals], dtype=float)
    vec = centers - position[None, :]
    dist = np.linalg.norm(vec, axis=1)
    nearest = int(np.argmin(dist))
    speed = np.linalg.norm(velocity)
    heading = velocity / max(speed, 1e-6)
    direction = vec[nearest] / max(dist[nearest], 1e-6)
    support = float(goals[nearest].get("support_fraction", 0.0))
    quality = goals[nearest].get("region_type", "")
    return np.asarray(
        [
            min(float(dist[nearest]), 100.0) / 100.0,
            float(np.dot(heading, direction)),
            float(np.cross(np.append(heading, 0.0), np.append(direction, 0.0))[2]),
            min(len(goals) / 8.0, 1.0),
            support,
            1.0 if "silver" in quality else 0.0,
            1.0 if "true" in quality or "gold" in quality else 0.0,
            1.0 if pack.get("route_corridors") else 0.0,
        ],
        dtype=float,
    )
