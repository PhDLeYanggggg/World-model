from __future__ import annotations

from typing import Dict, List

import numpy as np


def endpoint_clusters(endpoints: np.ndarray, max_goals: int = 6, seed: int = 17) -> List[Dict]:
    """Cluster training endpoints into inferred scene-level candidate goals.

    The resulting goals are inferred_scene_goal entries. They are built at the
    scene level from training endpoints and must not be treated as per-sample
    observed future goals.
    """
    points = np.asarray(endpoints, dtype=float)
    points = points[np.isfinite(points).all(axis=1)] if points.size else points.reshape(0, 2)
    if len(points) == 0:
        return []
    k = int(min(max_goals, max(1, round(np.sqrt(len(points))))))
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), size=k, replace=len(points) < k)].copy()
    for _ in range(25):
        d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        new_centers = centers.copy()
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                new_centers[idx] = points[mask].mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(d, axis=1)
    goals = []
    for idx in range(k):
        mask = labels == idx
        if not mask.any():
            continue
        cluster_points = points[mask]
        radius = float(max(np.percentile(np.linalg.norm(cluster_points - centers[idx], axis=1), 80), 0.5))
        goals.append(
            {
                "goal_id": f"inferred_goal_{idx}",
                "goal_type": "inferred_scene_goal",
                "center": [float(centers[idx, 0]), float(centers[idx, 1])],
                "radius": radius,
                "support_count": int(mask.sum()),
                "support_fraction": float(mask.mean()),
            }
        )
    goals.sort(key=lambda g: (-g["support_count"], g["goal_id"]))
    for idx, goal in enumerate(goals):
        goal["goal_id"] = f"inferred_goal_{idx}"
    return goals


def assign_goal(point: np.ndarray, goals: List[Dict]) -> int:
    if not goals:
        return -1
    centers = np.asarray([g["center"] for g in goals], dtype=float)
    return int(np.argmin(np.linalg.norm(centers - np.asarray(point, dtype=float), axis=1)))

