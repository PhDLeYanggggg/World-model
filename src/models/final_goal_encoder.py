from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class GoalEncoder:
    """Candidate-goal encoder. Goals must be train-only or annotated, never test endpoints."""

    def encode(self, last_positions: np.ndarray, goal_features: Dict[str, Any] | List[Any] | None) -> Dict[str, np.ndarray]:
        n = last_positions.shape[0]
        if not goal_features:
            return {"nearest_goal_distance": np.full(n, np.inf), "goal_available": np.zeros(n)}
        goals = goal_features.get("candidate_goals", goal_features) if isinstance(goal_features, dict) else goal_features
        coords = []
        for goal in goals:
            if isinstance(goal, dict):
                xy = goal.get("xy") or goal.get("center") or goal.get("centroid")
            else:
                xy = goal
            if xy is not None and len(xy) >= 2:
                coords.append([float(xy[0]), float(xy[1])])
        if not coords:
            return {"nearest_goal_distance": np.full(n, np.inf), "goal_available": np.zeros(n)}
        goal_arr = np.asarray(coords, dtype=np.float64)
        dists = np.linalg.norm(last_positions[:, None, :] - goal_arr[None, :, :], axis=2)
        return {"nearest_goal_distance": dists.min(axis=1), "goal_available": np.ones(n)}

