from __future__ import annotations

import math

import numpy as np

from src.physics.kinematics import clip_vectors, unit_vector
from src.physics.scene_geometry import SceneSpec, nearest_obstacle_vector


def local_density(state: np.ndarray, radius: float = 2.0) -> np.ndarray:
    densities = np.zeros(state.shape[0], dtype=np.float32)
    for i in range(state.shape[0]):
        d = np.linalg.norm(state[:, :2] - state[i, :2], axis=1)
        densities[i] = float(np.sum((d < radius) & (d > 0.0)))
    return densities


def compute_social_force_acceleration(
    state: np.ndarray,
    scene: SceneSpec,
    max_accel: float,
    goal_force_weight: float = 1.05,
    social_force_weight: float = 1.0,
    obstacle_force_weight: float = 1.0,
    boundary_force_weight: float = 1.0,
    damping: float = 0.10,
    use_goal_force: bool = True,
    use_social_force: bool = True,
    use_obstacle_force: bool = True,
    use_boundary_force: bool = True,
    repulsion_radius: float = 2.3,
) -> np.ndarray:
    n = state.shape[0]
    accel = np.zeros((n, 2), dtype=np.float32)
    positions = state[:, :2]
    velocities = state[:, 2:4]
    radii = state[:, 7]
    goals = state[:, 10:12]
    desired_speed = state[:, 8]
    density = local_density(state)

    for i in range(n):
        if use_goal_force and goal_force_weight != 0.0:
            goal_dir = unit_vector(goals[i] - positions[i])
            desired_v = goal_dir * desired_speed[i] * np.clip(1.0 - 0.07 * density[i], 0.35, 1.0)
            accel[i] += goal_force_weight * (desired_v - velocities[i])
        if damping != 0.0:
            accel[i] += -float(damping) * velocities[i]

        if use_social_force and social_force_weight != 0.0:
            for j in range(n):
                if i == j:
                    continue
                delta = positions[i] - positions[j]
                dist = max(1e-4, float(np.linalg.norm(delta)))
                gap = dist - float(radii[i] + radii[j])
                if gap > repulsion_radius:
                    continue
                normal = delta / dist
                closing = max(0.0, -float(np.dot(velocities[i] - velocities[j], normal)))
                strength = 0.55 * math.exp(-max(gap, -0.5) / 0.75) + 0.12 * closing
                accel[i] += normal * strength * social_force_weight

        if use_obstacle_force and obstacle_force_weight != 0.0 and bool(getattr(scene, "has_real_obstacles", True)):
            obs_vec, obs_dist, inside = nearest_obstacle_vector(tuple(positions[i]), scene)
            if inside:
                accel[i] += obs_vec * 2.2 * obstacle_force_weight
            elif obs_dist < 1.9:
                accel[i] += obs_vec * (1.1 * (1.9 - obs_dist)) * obstacle_force_weight

        x, y = positions[i]
        margin = 1.2
        if use_boundary_force and boundary_force_weight != 0.0 and bool(getattr(scene, "has_real_boundary", True)):
            if x < margin:
                accel[i, 0] += (margin - x) * boundary_force_weight
            if scene.width - x < margin:
                accel[i, 0] -= (margin - (scene.width - x)) * boundary_force_weight
            if y < margin:
                accel[i, 1] += (margin - y) * boundary_force_weight
            if scene.height - y < margin:
                accel[i, 1] -= (margin - (scene.height - y)) * boundary_force_weight

    return clip_vectors(accel, max_accel)
