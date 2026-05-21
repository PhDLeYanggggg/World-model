from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from src.features.world_state_features import stage3_agent_feature_vector
from src.physics.scene_geometry import SceneSpec, nearest_obstacle_vector
from src.physics.social_force import compute_social_force_acceleration, local_density


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.maximum(std, 1e-6)


def denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * np.maximum(std, 1e-6) + mean


class DeterministicResidualWorldModel(nn.Module):
    def __init__(self, entity_dim: int, neighbor_dim: int, obstacle_dim: int, latent_dim: int = 0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.entity_encoder = nn.Sequential(nn.Linear(entity_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.temporal = nn.GRU(64, 72, batch_first=True)
        self.neighbor_interaction_encoder = nn.Sequential(nn.Linear(neighbor_dim, 48), nn.GELU(), nn.Linear(48, 32), nn.GELU())
        self.obstacle_feature_encoder = nn.Sequential(nn.Linear(obstacle_dim, 32), nn.GELU(), nn.Linear(32, 24), nn.GELU())
        self.residual_dynamics_head = nn.Sequential(nn.Linear(72 + 32 + 24 + latent_dim, 96), nn.GELU(), nn.Linear(96, 2))
        self.uncertainty_head = nn.Sequential(nn.Linear(72 + 32 + 24 + latent_dim, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, entity_seq: torch.Tensor, neighbor: torch.Tensor, obstacle: torch.Tensor, latent: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, k, d = entity_seq.shape
        encoded = self.entity_encoder(entity_seq.reshape(b * k, d)).reshape(b, k, -1)
        _, hidden = self.temporal(encoded)
        h = torch.cat([hidden[-1], self.neighbor_interaction_encoder(neighbor), self.obstacle_feature_encoder(obstacle)], dim=-1)
        if self.latent_dim:
            if latent is None:
                latent = torch.zeros((b, self.latent_dim), dtype=h.dtype, device=h.device)
            h = torch.cat([h, latent], dim=-1)
        residual = self.residual_dynamics_head(h)
        log_std = torch.clamp(self.uncertainty_head(h), -3.0, 1.0)
        return residual, log_std


def build_feature_tensors(episodes: List[Dict], history_steps: int) -> Dict:
    entity_rows, neighbor_rows, obstacle_rows, targets = [], [], [], []
    current_rows, next_rows, base_accel_rows, scene_bounds_rows = [], [], [], []
    for episode in episodes:
        states, scene = episode["states"], episode["scene"]
        for t in range(history_steps - 1, states.shape[0] - 1):
            base = states[t]
            physics_accel = compute_social_force_acceleration(base, scene, float(np.max(base[:, 9])))
            residual_target = states[t + 1, :, 4:6] - physics_accel
            for i in range(base.shape[0]):
                entity_rows.append(entity_sequence_features(states[t - history_steps + 1 : t + 1], scene, i))
                neighbor_rows.append(neighbor_features(base, i))
                obstacle_rows.append(obstacle_features(base, scene, i))
                targets.append(residual_target[i])
                current_rows.append(base[i])
                next_rows.append(states[t + 1, i])
                base_accel_rows.append(physics_accel[i])
                scene_bounds_rows.append([scene.width, scene.height])
    entity = np.asarray(entity_rows, dtype=np.float32)
    neighbor = np.asarray(neighbor_rows, dtype=np.float32)
    obstacle = np.asarray(obstacle_rows, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    return {
        "entity": entity,
        "neighbor": neighbor,
        "obstacle": obstacle,
        "target": target,
        "current": np.asarray(current_rows, dtype=np.float32),
        "next": np.asarray(next_rows, dtype=np.float32),
        "base_accel": np.asarray(base_accel_rows, dtype=np.float32),
        "scene_bounds": np.asarray(scene_bounds_rows, dtype=np.float32),
    }


def fit_normalization(pack: Dict) -> Dict:
    return {
        "entity_mean": pack["entity"].mean(axis=(0, 1)),
        "entity_std": pack["entity"].std(axis=(0, 1)) + 1e-6,
        "neighbor_mean": pack["neighbor"].mean(axis=0),
        "neighbor_std": pack["neighbor"].std(axis=0) + 1e-6,
        "obstacle_mean": pack["obstacle"].mean(axis=0),
        "obstacle_std": pack["obstacle"].std(axis=0) + 1e-6,
        "target_mean": pack["target"].mean(axis=0),
        "target_std": pack["target"].std(axis=0) + 1e-6,
    }


def apply_normalization(pack: Dict, norm: Dict) -> Dict:
    return {
        "entity": normalize(pack["entity"], norm["entity_mean"], norm["entity_std"]),
        "neighbor": normalize(pack["neighbor"], norm["neighbor_mean"], norm["neighbor_std"]),
        "obstacle": normalize(pack["obstacle"], norm["obstacle_mean"], norm["obstacle_std"]),
        "target": normalize(pack["target"], norm["target_mean"], norm["target_std"]),
        "target_raw": pack["target"],
        "current": pack["current"],
        "next": pack["next"],
        "base_accel": pack["base_accel"],
        "scene_bounds": pack["scene_bounds"],
        "neighbor_raw": pack["neighbor"],
        "obstacle_raw": pack["obstacle"],
    }


def entity_sequence_features(history: np.ndarray, scene: SceneSpec, agent_index: int) -> np.ndarray:
    rows = []
    for frame in history:
        a = frame[agent_index]
        goal_vec = a[10:12] - a[0:2]
        dist_goal = float(np.linalg.norm(goal_vec))
        goal_dir = goal_vec / max(1e-6, dist_goal)
        obs_vec, obs_dist, inside = nearest_obstacle_vector(tuple(a[:2]), scene)
        boundary_dist = min(a[0], scene.width - a[0], a[1], scene.height - a[1])
        density = local_density(frame)[agent_index]
        rows.append(
            [
                a[0] / scene.width,
                a[1] / scene.height,
                a[2] / max(1e-6, a[8]),
                a[3] / max(1e-6, a[8]),
                a[4] / max(1e-6, a[9]),
                a[5] / max(1e-6, a[9]),
                a[6],
                a[7],
                min(dist_goal, 20.0) / 20.0,
                goal_dir[0],
                goal_dir[1],
                min(obs_dist, 6.0) / 6.0,
                min(boundary_dist, 6.0) / 6.0,
                density / 10.0,
                a[12] / 8.0,
                1.0 if inside else 0.0,
                *stage3_agent_feature_vector(frame, scene, agent_index).tolist(),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def neighbor_features(state: np.ndarray, i: int) -> np.ndarray:
    delta = state[:, :2] - state[i, :2]
    distances = np.linalg.norm(delta, axis=1)
    distances[i] = np.inf
    order = np.argsort(distances)[:6]
    valid = order[np.isfinite(distances[order])]
    if len(valid) == 0:
        return np.zeros(12, dtype=np.float32)
    rel = delta[valid]
    rel_v = state[valid, 2:4] - state[i, 2:4]
    gaps = distances[valid] - (state[valid, 7] + state[i, 7])
    nearest = valid[0]
    bearing = np.arctan2(delta[nearest, 1], delta[nearest, 0])
    return np.asarray(
        [
            rel[:, 0].mean() / 5.0,
            rel[:, 1].mean() / 5.0,
            distances[valid].mean() / 5.0,
            distances[nearest] / 5.0,
            rel_v[:, 0].mean() / 2.2,
            rel_v[:, 1].mean() / 2.2,
            gaps.min(),
            gaps.mean(),
            np.sin(bearing),
            np.cos(bearing),
            np.sum(distances < 2.0) / 10.0,
            np.sum(gaps < 0.3) / 10.0,
        ],
        dtype=np.float32,
    )


def obstacle_features(state: np.ndarray, scene: SceneSpec, i: int) -> np.ndarray:
    a = state[i]
    obs_vec, obs_dist, inside = nearest_obstacle_vector(tuple(a[:2]), scene)
    boundary = min(a[0], scene.width - a[0], a[1], scene.height - a[1])
    return np.asarray([obs_vec[0], obs_vec[1], min(obs_dist, 6.0) / 6.0, 1.0 if inside else 0.0, min(boundary, 6.0) / 6.0, scene.width / 50.0, scene.height / 50.0, len(scene.obstacles) / 6.0], dtype=np.float32)
