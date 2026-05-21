from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from src.physics.collision import min_gap_and_collisions
from src.physics.scene_geometry import SceneSpec, point_in_any_rect
from src.physics.social_force import local_density


SEMANTIC_LABELS = [
    "smooth_passage",
    "corridor_jam",
    "obstacle_detour",
    "group_split",
    "high_density_slowdown",
    "stalled",
    "collision_risk",
    "physically_invalid",
    "uncertain_multimodal",
]


def terminal_world_clustering(
    trajectories: np.ndarray,
    weights: np.ndarray,
    scene: SceneSpec,
    true_future: np.ndarray | None = None,
    true_event: str | None = None,
) -> Dict:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(1e-12, float(np.sum(weights)))
    features = [semantic_event_features(particle, scene) for particle in trajectories]
    labels = [semantic_label_from_features(item) for item in features]
    clusters: List[Dict] = []

    for cid, label in enumerate(sorted(set(labels), key=lambda item: SEMANTIC_LABELS.index(item) if item in SEMANTIC_LABELS else 99)):
        indexes = [i for i, item in enumerate(labels) if item == label]
        if not indexes:
            continue
        cluster_weights = weights[indexes]
        mass = float(np.sum(cluster_weights))
        representative = int(indexes[int(np.argmax(cluster_weights))])
        cluster_features = [features[i] for i in indexes]
        ade_values, fde_values = [], []
        if true_future is not None:
            h = min(100, true_future.shape[0] - 1, trajectories.shape[1] - 1)
            for index in indexes:
                err = np.linalg.norm(trajectories[index, 1 : h + 1, :, :2] - true_future[1 : h + 1, :, :2], axis=2)
                ade_values.append(float(np.mean(err)))
                fde_values.append(float(np.mean(np.linalg.norm(trajectories[index, h, :, :2] - true_future[h, :, :2], axis=1))))
        clusters.append(
            {
                "cluster_id": int(cid),
                "probability_mass": round(mass, 5),
                "semantic_label": label,
                "representative_trajectory_id": representative,
                "mean_ADE@100": round(float(np.mean(ade_values)), 4) if ade_values else None,
                "mean_FDE@100": round(float(np.mean(fde_values)), 4) if fde_values else None,
                "mean_collision_rate": round(float(np.mean([f["collision_rate"] for f in cluster_features])), 5),
                "mean_obstacle_violation_rate": round(float(np.mean([f["obstacle_violation_rate"] for f in cluster_features])), 5),
                "mean_boundary_violation_rate": round(float(np.mean([f["boundary_violation_rate"] for f in cluster_features])), 5),
                "mean_goal_reached_rate": round(float(np.mean([f["goal_reached_fraction"] for f in cluster_features])), 5),
                "mean_jam_duration": round(float(np.mean([f["jam_duration_fraction"] for f in cluster_features])), 5),
                "confidence": round(float(mass * (1.0 - np.mean([f["physical_violation_score"] for f in cluster_features]))), 5),
                "is_credible": bool(mass >= 0.08 and np.mean([f["physical_violation_score"] for f in cluster_features]) < 0.12),
                "explanation": naming_reason(label),
                "matches_true_event": bool(event_matches(label, true_event)) if true_event else None,
            }
        )

    clusters = sorted(clusters, key=lambda row: row["probability_mass"], reverse=True)
    diversity = cluster_diversity(clusters)
    top_label = clusters[0]["semantic_label"] if clusters else None
    return {
        "clusters": clusters,
        "cluster_diversity_score": round(diversity, 5),
        "top_cluster_label": top_label,
        "semantic_event_accuracy": float(event_matches(top_label, true_event)) if true_event else None,
        "semantic_diversity_note": "multi-branch sampling exists, but semantic diversity remains weak." if diversity < 0.35 else "semantic clusters are meaningfully separated.",
    }


def semantic_event_features(trajectory: np.ndarray, scene: SceneSpec) -> Dict:
    final = trajectory[-1]
    goal_dist = np.linalg.norm(final[:, :2] - final[:, 10:12], axis=1)
    goal_reached_fraction = float(np.mean(goal_dist < 1.4))
    speeds = np.linalg.norm(trajectory[:, :, 2:4], axis=2)
    jam_duration_fraction = float(np.mean(speeds < 0.22))
    mean_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))
    high_density_fraction = float(np.mean([np.mean(local_density(frame) >= 5.0) for frame in trajectory[::2]]))
    path_ratio = path_efficiency(trajectory)
    final_spread = float(np.std(final[:, 0]) + np.std(final[:, 1]))
    initial_spread = float(np.std(trajectory[0, :, 0]) + np.std(trajectory[0, :, 1]))
    split_flow = final_spread > max(10.0, initial_spread * 1.25)
    merged = final_spread < initial_spread * 0.65
    stalled = jam_duration_fraction > 0.46
    collision_frames = 0
    near_collision_frames = 0
    obstacle_frames = 0
    boundary_frames = 0
    min_gaps = []
    for frame in trajectory:
        min_gap, collisions = min_gap_and_collisions(frame)
        min_gaps.append(float(min_gap))
        if collisions:
            collision_frames += 1
        if min_gap < 0.12:
            near_collision_frames += 1
        obstacle = 0
        boundary = 0
        for agent in frame:
            radius = float(agent[7])
            if point_in_any_rect(tuple(agent[:2]), scene.obstacles, pad=radius):
                obstacle += 1
            if agent[0] < radius or agent[0] > scene.width - radius or agent[1] < radius or agent[1] > scene.height - radius:
                boundary += 1
        if obstacle:
            obstacle_frames += 1
        if boundary:
            boundary_frames += 1
    pass_time = first_reach_time(trajectory)
    smoothness = float(np.linalg.norm(np.diff(trajectory[:, :, 4:6], axis=0), axis=2).mean()) if trajectory.shape[0] > 1 else 0.0
    frames = max(1, trajectory.shape[0])
    physical_violation_score = min(
        1.0,
        collision_frames / frames
        + obstacle_frames / frames
        + boundary_frames / frames
        + max(0.0, max_speed - 2.6) / 3.0
        + max(0.0, smoothness - 3.5) / 8.0,
    )
    return {
        "goal_reached_fraction": goal_reached_fraction,
        "average_pass_time": pass_time,
        "jam_duration_fraction": jam_duration_fraction,
        "detour_score": float(path_ratio),
        "split_flow": float(split_flow),
        "merged_flow": float(merged),
        "high_density_fraction": high_density_fraction,
        "long_stall": float(stalled),
        "collision_rate": collision_frames / frames,
        "near_collision_rate": near_collision_frames / frames,
        "obstacle_violation_rate": obstacle_frames / frames,
        "boundary_violation_rate": boundary_frames / frames,
        "mean_speed_mps": mean_speed,
        "max_speed_mps": max_speed,
        "smoothness": smoothness,
        "physical_violation_score": physical_violation_score,
        "min_gap_m": float(np.min(min_gaps)) if min_gaps else 0.0,
        "final_spatial_spread": final_spread,
    }


def semantic_label_from_features(features: Dict) -> str:
    if features["obstacle_violation_rate"] > 0.02 or features["boundary_violation_rate"] > 0.02:
        return "physically_invalid"
    if features["long_stall"] > 0.5:
        return "stalled"
    if features["jam_duration_fraction"] > 0.32 and features["high_density_fraction"] > 0.08:
        return "corridor_jam"
    if features["detour_score"] > 1.48:
        return "obstacle_detour"
    if features["split_flow"] > 0.5:
        return "group_split"
    if features["high_density_fraction"] > 0.18:
        return "high_density_slowdown"
    if features["goal_reached_fraction"] > 0.45 and features["mean_speed_mps"] > 0.35:
        return "smooth_passage"
    if features["collision_rate"] > 0.01 or features["min_gap_m"] < 0.03 or features["near_collision_rate"] > 0.62:
        return "collision_risk"
    return "uncertain_multimodal"


def path_efficiency(trajectory: np.ndarray) -> float:
    path = np.linalg.norm(np.diff(trajectory[:, :, :2], axis=0), axis=2).sum(axis=0)
    straight = np.linalg.norm(trajectory[-1, :, :2] - trajectory[0, :, :2], axis=1)
    return float(np.mean(path / np.maximum(straight, 1.0)))


def first_reach_time(trajectory: np.ndarray) -> float:
    for t, frame in enumerate(trajectory):
        if np.mean(np.linalg.norm(frame[:, :2] - frame[:, 10:12], axis=1) < 1.4) > 0.5:
            return float(t)
    return float(trajectory.shape[0])


def cluster_diversity(clusters: List[Dict]) -> float:
    if len(clusters) <= 1:
        return 0.0
    probs = np.asarray([max(1e-12, cluster["probability_mass"]) for cluster in clusters], dtype=np.float64)
    probs /= probs.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy / max(1e-12, math.log(len(probs)))


def event_matches(predicted: str | None, true_event: str | None) -> bool:
    if predicted is None or true_event is None:
        return False
    equivalent = {
        "physically_corrected_collision_risk": "collision_risk",
        "collision-risk": "collision_risk",
        "congestion": "corridor_jam",
        "detour": "obstacle_detour",
        "split-flow": "group_split",
        "partial-flow": "uncertain_multimodal",
    }
    return equivalent.get(predicted, predicted) == equivalent.get(true_event, true_event)


def naming_reason(label: str) -> str:
    reasons = {
        "smooth_passage": "Most agents keep forward progress and a meaningful fraction reaches goals.",
        "corridor_jam": "Low-speed time and local density are both elevated, consistent with queueing.",
        "obstacle_detour": "Path length is substantially longer than the direct displacement.",
        "group_split": "Final spatial spread is larger than the initial spread, indicating split flow.",
        "high_density_slowdown": "The rollout spends time in dense regions without a full stop.",
        "stalled": "Agents remain below walking speed for a large fraction of the rollout.",
        "collision_risk": "Minimum gaps are small or collision projection was frequently needed.",
        "physically_invalid": "The rollout enters obstacle or boundary-violating regions.",
        "uncertain_multimodal": "No dominant semantic event passed the rule thresholds.",
    }
    return reasons.get(label, "Semantic label assigned by event-feature thresholds.")
