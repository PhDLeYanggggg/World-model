from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np

from src.physics.collision import min_gap_and_collisions
from src.physics.scene_geometry import SceneSpec, point_in_any_rect


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    peak = float(np.max(values))
    weights = np.exp(values - peak)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(values) / max(1, len(values))
    return weights / total


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    return float(1.0 / max(1e-12, np.sum(weights * weights)))


def systematic_resample_indexes(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    return np.searchsorted(cumulative, positions, side="right")


def trajectory_metrics(
    trajectories: np.ndarray,
    weights: np.ndarray,
    true_future: np.ndarray,
    scene: SceneSpec,
    horizons: Iterable[int],
    sigma_m: float = 1.25,
) -> Dict:
    particles = trajectories.shape[0]
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(1e-12, float(np.sum(weights)))
    weighted = np.tensordot(weights, trajectories, axes=(0, 0))
    out: Dict = {"horizons": {}, "branch_count": int(particles)}
    max_h = min(trajectories.shape[1] - 1, true_future.shape[0] - 1)

    for horizon in horizons:
        h = min(int(horizon), max_h)
        pred = weighted[: h + 1, :, :2]
        truth = true_future[: h + 1, :, :2]
        step_errors = np.linalg.norm(pred[1:] - truth[1:], axis=2)
        fde_agents = np.linalg.norm(pred[h] - truth[h], axis=1)
        particle_ade = np.mean(
            np.linalg.norm(trajectories[:, 1 : h + 1, :, :2] - truth[None, 1 : h + 1, :, :2], axis=3),
            axis=(1, 2),
        )
        particle_fde = np.mean(np.linalg.norm(trajectories[:, h, :, :2] - truth[None, h, :, :2], axis=2), axis=1)
        out["horizons"][str(horizon)] = {
            "evaluated_horizon": int(h),
            "ADE_m": round(float(np.mean(step_errors)), 4),
            "FDE_m": round(float(np.mean(fde_agents)), 4),
            "best_of_N_ADE_m": round(float(np.min(particle_ade)), 4),
            "best_of_N_FDE_m": round(float(np.min(particle_fde)), 4),
            "best_of_64_ADE_m": round(float(np.min(particle_ade)), 4),
            "best_of_64_FDE_m": round(float(np.min(particle_fde)), 4),
        }

    out.update(physical_violation_metrics(trajectories, scene))
    h100 = min(100, max_h)
    particle_fde_100 = np.mean(np.linalg.norm(trajectories[:, h100, :, :2] - true_future[None, h100, :, :2], axis=2), axis=1)
    out["coverage@64"] = round(float(np.mean(particle_fde_100 < 2.0)), 5)
    out["coverage_t100_fde_lt_2m"] = round(float(np.mean(particle_fde_100 < 2.0)), 5)
    out["NLL_endpoint_t100"] = round(endpoint_particle_nll(trajectories[:, h100, :, :2], weights, true_future[h100, :, :2], sigma_m), 4)
    return out


def physical_violation_metrics(trajectories: np.ndarray, scene: SceneSpec) -> Dict:
    particles, frames, agents, _ = trajectories.shape
    total_particle_frames = max(1, particles * frames)
    total_agent_frames = max(1, particles * frames * agents)
    collision_frames = 0
    obstacle_frames = 0
    boundary_frames = 0
    speed_violations = 0
    accel_violations = 0
    min_gaps: List[float] = []
    jerk_values: List[float] = []

    for particle in trajectories:
        if particle.shape[0] > 1:
            jerk_values.append(float(np.linalg.norm(np.diff(particle[:, :, 4:6], axis=0), axis=2).mean()))
        for frame in particle:
            min_gap, collisions = min_gap_and_collisions(frame)
            min_gaps.append(float(min_gap))
            if collisions > 0 or min_gap < -1e-4:
                collision_frames += 1
            obstacle = 0
            boundary = 0
            for agent in frame:
                radius = float(agent[7])
                if agent[0] < radius or agent[0] > scene.width - radius or agent[1] < radius or agent[1] > scene.height - radius:
                    boundary += 1
                if point_in_any_rect(tuple(agent[:2]), scene.obstacles, pad=radius):
                    obstacle += 1
                if np.linalg.norm(agent[2:4]) > float(agent[8]) + 1e-4:
                    speed_violations += 1
                if np.linalg.norm(agent[4:6]) > float(agent[9]) + 1e-4:
                    accel_violations += 1
            if obstacle:
                obstacle_frames += 1
            if boundary:
                boundary_frames += 1

    return {
        "collision_violation_rate": round(collision_frames / total_particle_frames, 5),
        "obstacle_violation_rate": round(obstacle_frames / total_particle_frames, 5),
        "boundary_violation_rate": round(boundary_frames / total_particle_frames, 5),
        "max_speed_violation_rate": round(speed_violations / total_agent_frames, 5),
        "acceleration_violation_rate": round(accel_violations / total_agent_frames, 5),
        "trajectory_smoothness": round(float(np.mean(jerk_values)) if jerk_values else 0.0, 4),
        "smoothness_score": round(float(np.mean(jerk_values)) if jerk_values else 0.0, 4),
        "min_gap_m": round(float(np.min(min_gaps)) if min_gaps else 0.0, 4),
    }


def endpoint_particle_nll(particle_endpoints: np.ndarray, weights: np.ndarray, truth: np.ndarray, sigma: float = 1.25) -> float:
    log_probs = []
    norm_const = -math.log(2 * math.pi * sigma * sigma)
    for i in range(truth.shape[0]):
        d2 = np.sum((particle_endpoints[:, i, :] - truth[i]) ** 2, axis=1)
        components = np.log(np.maximum(weights, 1e-12)) + norm_const - 0.5 * d2 / (sigma * sigma)
        peak = np.max(components)
        log_probs.append(float(peak + math.log(float(np.sum(np.exp(components - peak))))))
    return float(-np.mean(log_probs))


def aggregate_metric_dicts(results: List[Dict], horizons: Iterable[int]) -> Dict:
    out: Dict = {"horizons": {}}
    if not results:
        return out
    for horizon in horizons:
        key = str(horizon)
        out["horizons"][key] = {}
        hkeys = sorted(results[0]["horizons"][key].keys())
        for metric in hkeys:
            values = [r["horizons"][key][metric] for r in results if isinstance(r["horizons"][key].get(metric), (int, float))]
            out["horizons"][key][metric] = round(float(np.mean(values)), 4) if values else None
    scalar_keys = [
        "branch_count",
        "collision_violation_rate",
        "obstacle_violation_rate",
        "boundary_violation_rate",
        "max_speed_violation_rate",
        "acceleration_violation_rate",
        "trajectory_smoothness",
        "smoothness_score",
        "min_gap_m",
        "coverage@64",
        "coverage_t100_fde_lt_2m",
        "NLL_endpoint_t100",
    ]
    for metric in scalar_keys:
        values = [r[metric] for r in results if isinstance(r.get(metric), (int, float))]
        out[metric] = round(float(np.mean(values)), 5) if values else None
    return out
