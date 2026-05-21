from __future__ import annotations

import math
from typing import Dict

import numpy as np

from src.physics.scene_geometry import SceneSpec, nearest_obstacle_vector


def pairwise_interaction_features(state: np.ndarray, index: int) -> Dict[str, float]:
    agent = state[index]
    positions = state[:, :2]
    velocities = state[:, 2:4]
    radii = state[:, 7]
    delta = positions - agent[:2]
    distances = np.linalg.norm(delta, axis=1)
    distances[index] = np.inf
    nearest = int(np.argmin(distances))
    if not np.isfinite(distances[nearest]):
        return {
            "nearest_neighbor_distance_m": 99.0,
            "gap_m": 99.0,
            "closing_speed_mps": 0.0,
            "time_to_collision_s": 99.0,
            "front_density": 0.0,
            "rear_density": 0.0,
            "left_density": 0.0,
            "right_density": 0.0,
        }
    rel = state[nearest, :2] - agent[:2]
    rel_dist = max(1e-6, float(np.linalg.norm(rel)))
    normal = rel / rel_dist
    rel_vel = state[nearest, 2:4] - agent[2:4]
    closing_speed = max(0.0, -float(np.dot(rel_vel, normal)))
    gap = rel_dist - float(radii[index] + radii[nearest])
    ttc = gap / closing_speed if closing_speed > 1e-5 and gap > 0 else (0.0 if gap <= 0 else 99.0)
    heading = heading_vector(agent)
    left = np.asarray([-heading[1], heading[0]], dtype=np.float32)
    front = rear = left_count = right_count = 0.0
    for j in range(state.shape[0]):
        if j == index:
            continue
        d = float(np.linalg.norm(delta[j]))
        if d > 3.0 or d < 1e-6:
            continue
        direction = delta[j] / d
        if float(np.dot(direction, heading)) > 0.35:
            front += 1.0
        elif float(np.dot(direction, heading)) < -0.35:
            rear += 1.0
        if float(np.dot(direction, left)) > 0.35:
            left_count += 1.0
        elif float(np.dot(direction, left)) < -0.35:
            right_count += 1.0
    sector_area = math.pi * 3.0 * 3.0 / 2.0
    return {
        "nearest_neighbor_distance_m": rel_dist,
        "gap_m": gap,
        "closing_speed_mps": closing_speed,
        "time_to_collision_s": min(99.0, ttc),
        "front_density": front / sector_area,
        "rear_density": rear / sector_area,
        "left_density": left_count / sector_area,
        "right_density": right_count / sector_area,
    }


def scene_context_features(state: np.ndarray, scene: SceneSpec, index: int) -> Dict[str, float]:
    agent = state[index]
    obs_vec, obs_dist, inside = nearest_obstacle_vector(tuple(agent[:2]), scene)
    tangent = np.asarray([-obs_vec[1], obs_vec[0]], dtype=np.float32)
    boundary_dist = min(agent[0], scene.width - agent[0], agent[1], scene.height - agent[1])
    exit_distances = [float(np.linalg.norm(agent[:2] - np.asarray(exit_xy, dtype=np.float32))) for exit_xy in scene.exits.values()]
    nearest_exit = min(exit_distances) if exit_distances else 99.0
    bottleneck = bottleneck_score(agent[:2], scene)
    return {
        "obstacle_tangent_x": float(tangent[0]),
        "obstacle_tangent_y": float(tangent[1]),
        "inside_obstacle": float(inside),
        "boundary_distance_m": float(boundary_dist),
        "nearest_exit_distance_m": float(nearest_exit),
        "bottleneck_score": float(bottleneck),
        "scene_aspect_ratio": float(scene.width / max(1e-6, scene.height)),
    }


def intent_features(state: np.ndarray, index: int) -> Dict[str, float]:
    agent = state[index]
    velocity_dir = heading_vector(agent)
    goal_vec = agent[10:12] - agent[:2]
    goal_dist = float(np.linalg.norm(goal_vec))
    goal_dir = goal_vec / max(1e-6, goal_dist)
    speed = float(np.linalg.norm(agent[2:4]))
    accel = float(np.linalg.norm(agent[4:6]))
    alignment = float(np.dot(velocity_dir, goal_dir))
    dwell = float(speed < 0.18)
    heading_rate_proxy = float(abs(np.sin(agent[6])) * accel / max(0.2, speed))
    return {
        "goal_distance_m": goal_dist,
        "velocity_goal_alignment": alignment,
        "speed_ratio": speed / max(1e-6, float(agent[8])),
        "accel_ratio": accel / max(1e-6, float(agent[9])),
        "dwell_flag": dwell,
        "heading_rate_proxy": min(6.0, heading_rate_proxy),
    }


def stage3_agent_feature_vector(state: np.ndarray, scene: SceneSpec, index: int) -> np.ndarray:
    interaction = pairwise_interaction_features(state, index)
    scene_features = scene_context_features(state, scene, index)
    intent = intent_features(state, index)
    values = [
        min(interaction["nearest_neighbor_distance_m"], 10.0) / 10.0,
        np.clip(interaction["gap_m"], -1.0, 5.0) / 5.0,
        min(interaction["closing_speed_mps"], 4.0) / 4.0,
        min(interaction["time_to_collision_s"], 10.0) / 10.0,
        interaction["front_density"],
        interaction["rear_density"],
        interaction["left_density"],
        interaction["right_density"],
        scene_features["obstacle_tangent_x"],
        scene_features["obstacle_tangent_y"],
        min(scene_features["boundary_distance_m"], 8.0) / 8.0,
        min(scene_features["nearest_exit_distance_m"], 30.0) / 30.0,
        scene_features["bottleneck_score"],
        min(intent["goal_distance_m"], 30.0) / 30.0,
        intent["velocity_goal_alignment"],
        intent["speed_ratio"],
        intent["accel_ratio"],
        intent["dwell_flag"],
        intent["heading_rate_proxy"] / 6.0,
    ]
    return np.asarray(values, dtype=np.float32)


def heading_vector(agent: np.ndarray) -> np.ndarray:
    vel = agent[2:4]
    speed = float(np.linalg.norm(vel))
    if speed > 1e-5:
        return (vel / speed).astype(np.float32)
    goal = agent[10:12] - agent[:2]
    norm = float(np.linalg.norm(goal))
    if norm > 1e-5:
        return (goal / norm).astype(np.float32)
    return np.asarray([1.0, 0.0], dtype=np.float32)


def bottleneck_score(point: np.ndarray, scene: SceneSpec) -> float:
    x, y = float(point[0]), float(point[1])
    vertical_clearance = min(y, scene.height - y)
    horizontal_clearance = min(x, scene.width - x)
    for rect in scene.obstacles:
        if rect.x1 <= x <= rect.x2:
            vertical_clearance = min(vertical_clearance, abs(y - rect.y1), abs(y - rect.y2))
        if rect.y1 <= y <= rect.y2:
            horizontal_clearance = min(horizontal_clearance, abs(x - rect.x1), abs(x - rect.x2))
    clearance = min(vertical_clearance, horizontal_clearance)
    return float(np.clip(1.0 - clearance / 5.0, 0.0, 1.0))
