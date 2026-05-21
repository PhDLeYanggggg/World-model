from __future__ import annotations

import numpy as np

from src.physics.collision import project_collisions
from src.physics.constraints import project_scene_constraints
from src.physics.kinematics import heading, integrate_state
from src.physics.scene_geometry import SceneSpec
from src.physics.social_force import compute_social_force_acceleration


def _maybe_project(current: np.ndarray, scene: SceneSpec, use_collision_projection: bool, use_scene_constraints: bool) -> np.ndarray:
    if use_collision_projection:
        current, _ = project_collisions(current)
    if use_scene_constraints:
        current, _ = project_scene_constraints(current, scene)
    return current


def constant_velocity_rollout(
    history: np.ndarray,
    scene: SceneSpec,
    horizon: int,
    dt: float,
    max_speed: float,
    max_accel: float,
    use_collision_projection: bool = True,
    use_scene_constraints: bool = True,
) -> np.ndarray:
    out = np.zeros((horizon + 1, history.shape[1], history.shape[2]), dtype=np.float32)
    out[0] = history[-1]
    current = history[-1].copy()
    for t in range(1, horizon + 1):
        current = current.copy()
        current[:, 0:2] += current[:, 2:4] * dt
        current = _maybe_project(current, scene, use_collision_projection, use_scene_constraints)
        out[t] = current
    return out


def constant_acceleration_rollout(
    history: np.ndarray,
    scene: SceneSpec,
    horizon: int,
    dt: float,
    max_speed: float,
    max_accel: float,
    use_collision_projection: bool = True,
    use_scene_constraints: bool = True,
) -> np.ndarray:
    out = np.zeros((horizon + 1, history.shape[1], history.shape[2]), dtype=np.float32)
    out[0] = history[-1]
    current = history[-1].copy()
    for t in range(1, horizon + 1):
        current = current.copy()
        current[:, 0:2] += current[:, 2:4] * dt + 0.5 * current[:, 4:6] * dt * dt
        current[:, 2:4] = current[:, 2:4] + current[:, 4:6] * dt
        current[:, 6] = heading(current[:, 2], current[:, 3])
        current = _maybe_project(current, scene, use_collision_projection, use_scene_constraints)
        out[t] = current
    return out


def damped_velocity_rollout(
    history: np.ndarray,
    scene: SceneSpec,
    horizon: int,
    dt: float,
    max_speed: float,
    max_accel: float,
    damping: float = 0.98,
    use_collision_projection: bool = True,
    use_scene_constraints: bool = True,
) -> np.ndarray:
    out = np.zeros((horizon + 1, history.shape[1], history.shape[2]), dtype=np.float32)
    out[0] = history[-1]
    current = history[-1].copy()
    for t in range(1, horizon + 1):
        current = current.copy()
        current[:, 2:4] *= float(damping)
        current[:, 4:6] = (current[:, 2:4] - out[t - 1, :, 2:4]) / max(dt, 1e-6)
        current[:, 0:2] += current[:, 2:4] * dt
        current[:, 6] = heading(current[:, 2], current[:, 3])
        current = _maybe_project(current, scene, use_collision_projection, use_scene_constraints)
        out[t] = current
    return out


def constant_turn_rate_velocity_rollout(
    history: np.ndarray,
    scene: SceneSpec,
    horizon: int,
    dt: float,
    max_speed: float,
    max_accel: float,
    use_collision_projection: bool = True,
    use_scene_constraints: bool = True,
) -> np.ndarray:
    out = np.zeros((horizon + 1, history.shape[1], history.shape[2]), dtype=np.float32)
    out[0] = history[-1]
    current = history[-1].copy()
    if history.shape[0] >= 2:
        dtheta = np.angle(np.exp(1j * (history[-1, :, 6] - history[-2, :, 6])))
    else:
        dtheta = np.zeros(history.shape[1], dtype=np.float32)
    speed = np.linalg.norm(current[:, 2:4], axis=1)
    theta = current[:, 6].copy()
    for t in range(1, horizon + 1):
        current = current.copy()
        theta = theta + dtheta
        current[:, 2] = speed * np.cos(theta)
        current[:, 3] = speed * np.sin(theta)
        current[:, 0:2] += current[:, 2:4] * dt
        current[:, 6] = theta
        current = _maybe_project(current, scene, use_collision_projection, use_scene_constraints)
        out[t] = current
    return out


def identity_hand_physics_rollout(
    history: np.ndarray,
    scene: SceneSpec,
    horizon: int,
    dt: float,
    max_speed: float,
    max_accel: float,
    use_collision_projection: bool = True,
    use_scene_constraints: bool = True,
) -> np.ndarray:
    return constant_velocity_rollout(history, scene, horizon, dt, max_speed, max_accel, use_collision_projection, use_scene_constraints)


def hand_physics_rollout(
    history: np.ndarray,
    scene: SceneSpec,
    horizon: int,
    dt: float,
    max_speed: float,
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
    use_collision_projection: bool = True,
    use_scene_constraints: bool = True,
) -> np.ndarray:
    out = np.zeros((horizon + 1, history.shape[1], history.shape[2]), dtype=np.float32)
    out[0] = history[-1]
    current = history[-1].copy()
    for t in range(1, horizon + 1):
        accel = compute_social_force_acceleration(
            current,
            scene,
            max_accel,
            goal_force_weight=goal_force_weight,
            social_force_weight=social_force_weight,
            obstacle_force_weight=obstacle_force_weight,
            boundary_force_weight=boundary_force_weight,
            damping=damping,
            use_goal_force=use_goal_force,
            use_social_force=use_social_force,
            use_obstacle_force=use_obstacle_force,
            use_boundary_force=use_boundary_force,
        )
        current = integrate_state(current, accel, dt, max_speed)
        current = _maybe_project(current, scene, use_collision_projection, use_scene_constraints)
        out[t] = current
    return out


def tuned_hand_physics_rollout(history: np.ndarray, scene: SceneSpec, horizon: int, dt: float, max_speed: float, max_accel: float) -> np.ndarray:
    has_scene = bool(getattr(scene, "has_real_boundary", False) or getattr(scene, "has_real_obstacles", False) or getattr(scene, "has_real_exits", False))
    return hand_physics_rollout(
        history,
        scene,
        horizon,
        dt,
        max_speed,
        max_accel,
        goal_force_weight=0.0,
        social_force_weight=0.0,
        obstacle_force_weight=0.0,
        boundary_force_weight=0.0,
        damping=0.0,
        use_goal_force=False,
        use_social_force=False,
        use_obstacle_force=False,
        use_boundary_force=False,
        use_collision_projection=True,
        use_scene_constraints=has_scene,
    )
