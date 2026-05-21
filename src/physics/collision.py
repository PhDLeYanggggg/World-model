from __future__ import annotations

import numpy as np


def min_gap_and_collisions(state: np.ndarray) -> tuple[float, int]:
    min_gap = 999.0
    collisions = 0
    n = state.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            gap = float(np.linalg.norm(state[i, :2] - state[j, :2]) - (state[i, 7] + state[j, 7]))
            min_gap = min(min_gap, gap)
            if gap < -1e-4:
                collisions += 1
    return min_gap, collisions


def project_collisions(state: np.ndarray, iterations: int = 3, comfort_margin: float = 0.02) -> tuple[np.ndarray, dict]:
    out = state.copy()
    cost = 0.0
    attempted = 0
    max_penetration = 0.0
    n = out.shape[0]
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = out[j, :2] - out[i, :2]
                dist = max(1e-6, float(np.linalg.norm(delta)))
                penetration = float(out[i, 7] + out[j, 7] + comfort_margin - dist)
                if penetration <= 0:
                    continue
                normal = delta / dist
                out[i, :2] -= normal * penetration * 0.5
                out[j, :2] += normal * penetration * 0.5
                attempted += 1
                cost += penetration
                max_penetration = max(max_penetration, penetration)
                moved = True
        if not moved:
            break
    min_gap, collisions = min_gap_and_collisions(out)
    return out, {
        "attempted_collision_count": attempted,
        "collision_count": collisions,
        "collision_projection_cost": float(cost),
        "min_gap": float(min_gap),
        "max_penetration": float(max_penetration),
    }
