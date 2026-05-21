from __future__ import annotations

import numpy as np

from src.physics.scene_geometry import SceneSpec, point_in_any_rect, push_out_rect


def project_scene_constraints(state: np.ndarray, scene: SceneSpec) -> tuple[np.ndarray, dict]:
    out = state.copy()
    boundary_violation = 0.0
    obstacle_violation = 0
    obstacle_cost = 0.0
    use_boundary = bool(getattr(scene, "has_real_boundary", True))
    use_obstacles = bool(getattr(scene, "has_real_obstacles", True))
    for i in range(out.shape[0]):
        r = float(out[i, 7])
        before = out[i, :2].copy()
        if use_boundary:
            out[i, 0] = float(np.clip(out[i, 0], r, scene.width - r))
            out[i, 1] = float(np.clip(out[i, 1], r, scene.height - r))
            boundary_violation += float(np.linalg.norm(out[i, :2] - before))
        if use_obstacles:
            for rect in scene.obstacles:
                if point_in_any_rect(tuple(out[i, :2]), [rect], pad=r):
                    obstacle_violation += 1
                    corrected = push_out_rect(tuple(out[i, :2]), rect, r + 0.04)
                    obstacle_cost += float(np.linalg.norm(out[i, :2] - np.asarray(corrected)))
                    out[i, :2] = corrected
    return out, {
        "obstacle_violation_count": obstacle_violation,
        "boundary_violation": float(boundary_violation),
        "obstacle_projection_cost": float(obstacle_cost),
    }


def violation_counts(state: np.ndarray, scene: SceneSpec, max_speed: float, max_accel: float) -> dict:
    use_boundary = bool(getattr(scene, "has_real_boundary", True))
    use_obstacles = bool(getattr(scene, "has_real_obstacles", True))
    obstacle = sum(point_in_any_rect(tuple(agent[:2]), scene.obstacles, pad=float(agent[7])) for agent in state) if use_obstacles else 0
    boundary = sum(agent[0] < agent[7] or agent[0] > scene.width - agent[7] or agent[1] < agent[7] or agent[1] > scene.height - agent[7] for agent in state) if use_boundary else 0
    speed = int(np.sum(np.linalg.norm(state[:, 2:4], axis=1) > max_speed + 1e-4))
    accel = int(np.sum(np.linalg.norm(state[:, 4:6], axis=1) > max_accel + 1e-4))
    return {"obstacle": int(obstacle), "boundary": int(boundary), "speed": speed, "accel": accel}
