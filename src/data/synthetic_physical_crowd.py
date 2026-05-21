from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.physics.collision import min_gap_and_collisions, project_collisions
from src.physics.constraints import project_scene_constraints, violation_counts
from src.physics.kinematics import heading, integrate_state
from src.physics.scene_geometry import SceneSpec, point_in_any_rect
from src.physics.social_force import compute_social_force_acceleration, local_density


STATE_COLUMNS = [
    "position_x",
    "position_y",
    "velocity_x",
    "velocity_y",
    "acceleration_x",
    "acceleration_y",
    "heading",
    "body_radius",
    "max_speed",
    "max_acceleration",
    "goal_x",
    "goal_y",
    "group_id",
    "active",
    "reached_goal",
    "collision_count",
    "obstacle_violation",
    "boundary_violation",
]


def simulate_episode(scene: SceneSpec, episode_id: int, split: str, cfg: Dict, rng: np.random.Generator) -> Dict:
    world = cfg["world"]
    frames = int(world.get("quick_frames", world["frames"]))
    max_agents = int(world.get("quick_max_agents", world["max_agents"]))
    n_agents = int(rng.integers(world["min_agents"], max_agents + 1))
    dt = float(world["dt"])
    max_speed = float(world["max_speed_mps"])
    max_accel = float(world["max_accel_mps2"])

    states = np.zeros((frames, n_agents, len(STATE_COLUMNS)), dtype=np.float32)
    pause = np.zeros(n_agents, dtype=np.int32)
    goal_changes = np.zeros(n_agents, dtype=np.int32)
    path_length = np.zeros(n_agents, dtype=np.float32)

    for i in range(n_agents):
        x, y, gx, gy = sample_agent(scene, states[0, :i], rng, world)
        radius = float(rng.uniform(world["body_radius_min_m"], world["body_radius_max_m"]))
        desired = float(rng.uniform(0.8, max_speed * 0.78))
        states[0, i] = [x, y, 0, 0, 0, 0, 0, radius, desired, max_accel, gx, gy, i % 4, 1, 0, 0, 0, 0]

    for t in range(frames - 1):
        current = states[t].copy()
        for i in range(n_agents):
            if np.linalg.norm(current[i, :2] - current[i, 10:12]) < 1.2:
                current[i, 14] = 1.0
                if rng.random() < 0.30:
                    gx, gy = random_goal(scene, rng)
                    current[i, 10:12] = [gx, gy]
                    current[i, 14] = 0.0
                    goal_changes[i] += 1
            if pause[i] > 0:
                pause[i] -= 1
                current[i, 8] = min(current[i, 8], 0.12)
            elif rng.random() < 0.006:
                pause[i] = int(rng.integers(4, 16))
            if rng.random() < 0.003:
                gx, gy = random_goal(scene, rng)
                current[i, 10:12] = [gx, gy]
                current[i, 14] = 0.0
                goal_changes[i] += 1

        accel = compute_social_force_acceleration(current, scene, max_accel)
        accel += rng.normal(0.0, 0.04, size=accel.shape).astype(np.float32)
        next_state = integrate_state(current, accel, dt, max_speed)
        after_collision, cinfo = project_collisions(next_state)
        after_scene, sinfo = project_scene_constraints(after_collision, scene)
        after_scene[:, 4:6] = (after_scene[:, 2:4] - current[:, 2:4]) / dt
        after_scene[:, 6] = heading(after_scene[:, 2], after_scene[:, 3])
        after_scene[:, 15] = cinfo["collision_count"]
        after_scene[:, 16] = sinfo["obstacle_violation_count"]
        after_scene[:, 17] = sinfo["boundary_violation"]
        path_length += np.linalg.norm(after_scene[:, :2] - current[:, :2], axis=1)
        states[t + 1] = after_scene

    meta = {
        "episode_id": int(episode_id),
        "split": split,
        "scene_name": scene.name,
        "event_label": classify_episode(states, scene),
        "frames": frames,
        "agents": n_agents,
        "state_columns": STATE_COLUMNS,
        "goal_changes": int(goal_changes.sum()),
        "mean_path_length_m": round(float(path_length.mean()), 3),
    }
    return {"meta": meta, "states": states}


def sample_agent(scene: SceneSpec, existing: np.ndarray, rng: np.random.Generator, world: Dict) -> Tuple[float, float, float, float]:
    weights = np.asarray([r["weight"] for r in scene.spawn_regions], dtype=float)
    weights /= weights.sum()
    for _ in range(500):
        region = scene.spawn_regions[int(rng.choice(len(scene.spawn_regions), p=weights))]
        x1, y1, x2, y2 = region["rect"]
        x, y = float(rng.uniform(x1, x2)), float(rng.uniform(y1, y2))
        if point_in_any_rect((x, y), scene.obstacles, pad=0.4):
            continue
        if len(existing):
            d = np.linalg.norm(existing[:, :2] - np.asarray([x, y]), axis=1)
            if np.any(d < existing[:, 7] + 0.55):
                continue
        gx, gy = scene.exits[region["goal"]]
        return x, y, float(gx + rng.normal(0, 0.45)), float(gy + rng.normal(0, 0.45))
    gx, gy = scene.exits["east"]
    return 2.0, scene.height * 0.5, gx, gy


def random_goal(scene: SceneSpec, rng: np.random.Generator) -> tuple[float, float]:
    key = rng.choice(list(scene.exits.keys()))
    gx, gy = scene.exits[str(key)]
    return float(gx + rng.normal(0, 0.4)), float(gy + rng.normal(0, 0.4))


def classify_episode(states: np.ndarray, scene: SceneSpec) -> str:
    final = states[-1]
    reached = float(np.mean(final[:, 14] > 0.5))
    speeds = np.linalg.norm(states[:, :, 2:4], axis=2)
    stopped = float(np.mean(speeds < 0.18))
    density = float(np.mean([np.mean(local_density(frame)) for frame in states[::5]]))
    min_gap = min(min_gap_and_collisions(frame)[0] for frame in states[::5])
    path_ratio = path_efficiency(states)
    spread = float(np.std(final[:, 0]) + np.std(final[:, 1]))
    if min_gap < -0.02:
        return "physically_corrected_collision_risk"
    if scene.event_hint == "smooth_passage" and stopped < 0.38 and density < 2.6:
        return "smooth_passage" if reached > 0.25 else "high_density_slowdown"
    if scene.event_hint in {"corridor_jam"} or (stopped > 0.30 and density > 1.5):
        return "corridor_jam"
    if scene.event_hint == "obstacle_detour" or path_ratio > 1.42:
        return "obstacle_detour"
    if scene.event_hint == "group_split" or spread > 13:
        return "group_split"
    if reached > 0.65:
        return "smooth_passage"
    if stopped > 0.45:
        return "stalled"
    return "high_density_slowdown"


def path_efficiency(states: np.ndarray) -> float:
    path = np.linalg.norm(np.diff(states[:, :, :2], axis=0), axis=2).sum(axis=0)
    straight = np.linalg.norm(states[-1, :, :2] - states[0, :, :2], axis=1)
    return float(np.mean(path / np.maximum(straight, 1.0)))


def save_episode(path: Path, episode: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, states=episode["states"], meta=json.dumps(episode["meta"]))
