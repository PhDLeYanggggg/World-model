#!/usr/bin/env python3
"""Stage 2: learned 2.5D crowd physics world model.

This script deliberately separates three things:

1. SyntheticPhysicalCrowd2.5D: a controlled long-horizon environment with true
   t+100 labels, goals, body radii, scene geometry, and physical diagnostics.
2. A neural residual transition model: hand-coded physics provides a base
   acceleration; a learned GRU + interaction encoder predicts residual
   acceleration in world coordinates.
3. SMC rollout comparison: hand-coded physics proposal, learned neural proposal,
   and hybrid physics + neural residual proposal.

The output report is intentionally conservative. It does not claim that
AerialMPT t+100 is measurable, because the selected real scene is too short.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "stage2_learned_world_model"
DATASET_DIR = OUT_ROOT / "synthetic_dataset"
MODEL_DIR = OUT_ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"

DT = 0.4
EPISODE_FRAMES = 160
K_HISTORY = 6
HORIZONS = [1, 10, 25, 50, 100]
MAX_SPEED = 2.15
MAX_ACCEL = 2.9
MAX_AGENTS = 50
PARTICLES = 64
DEVICE = torch.device("cpu")
EVAL_TEST_EPISODES = 4

STATE_COLUMNS = [
    "x",
    "y",
    "vx",
    "vy",
    "ax",
    "ay",
    "radius",
    "goal_x",
    "goal_y",
    "desired_speed",
    "active",
]


@dataclass
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str = "obstacle"


@dataclass
class SceneSpec:
    name: str
    width: float
    height: float
    obstacles: List[Rect]
    exits: Dict[str, Tuple[float, float]]
    spawn_regions: List[Dict]
    event_hint: str


class ResidualCrowdModel(nn.Module):
    def __init__(self, entity_dim: int, neighbor_dim: int, obstacle_dim: int, latent_dim: int = 4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.temporal = nn.GRU(64, 72, batch_first=True)
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_dim, 48),
            nn.GELU(),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(obstacle_dim, 32),
            nn.GELU(),
            nn.Linear(32, 24),
            nn.GELU(),
        )
        self.transition_head = nn.Sequential(
            nn.Linear(72 + 32 + 24 + latent_dim, 96),
            nn.GELU(),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        self.log_std = nn.Parameter(torch.tensor([-1.2, -1.2], dtype=torch.float32))

    def forward(self, entity_seq: torch.Tensor, neighbor: torch.Tensor, obstacle: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        b, k, d = entity_seq.shape
        encoded = self.entity_encoder(entity_seq.reshape(b * k, d)).reshape(b, k, -1)
        _, hidden = self.temporal(encoded)
        temporal = hidden[-1]
        neighbor_h = self.neighbor_encoder(neighbor)
        obstacle_h = self.obstacle_encoder(obstacle)
        return self.transition_head(torch.cat([temporal, neighbor_h, obstacle_h, z], dim=-1))


def main() -> None:
    seed_everything(123)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    scenes = make_scene_templates()
    dataset = generate_or_load_dataset(scenes)
    dataset_summary = summarize_dataset(dataset)
    print("dataset", json.dumps(dataset_summary, indent=2), flush=True)

    model_path = MODEL_DIR / "neural_residual_world_model.pt"
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        model = ResidualCrowdModel(
            entity_dim=int(checkpoint["entity_dim"]),
            neighbor_dim=int(checkpoint["neighbor_dim"]),
            obstacle_dim=int(checkpoint["obstacle_dim"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        normalization = checkpoint["normalization"]
        training_summary = checkpoint["training_summary"]
        print(f"loaded existing residual model from {model_path}", flush=True)
    else:
        train_pack = build_training_tensors(dataset, split="train")
        val_pack = build_training_tensors(dataset, split="val")
        model, training_summary = train_residual_model(train_pack, val_pack)
        normalization = train_pack["normalization"]
        torch.save(
            {
                "state_dict": model.state_dict(),
                "normalization": normalization,
                "entity_dim": train_pack["entity"].shape[-1],
                "neighbor_dim": train_pack["neighbor"].shape[-1],
                "obstacle_dim": train_pack["obstacle"].shape[-1],
                "state_columns": STATE_COLUMNS,
                "training_summary": training_summary,
            },
            model_path,
        )

    evaluation = evaluate_smc_proposals(dataset, model, normalization)
    cluster_summary = build_terminal_cluster_report(evaluation)
    aerial_summary = load_aerialmpt_stage1_summary()

    report = {
        "stage": "stage2_learned_2_5d_world_model",
        "runtime_s": round(time.time() - t0, 3),
        "dataset": dataset_summary,
        "model": {
            "architecture": {
                "entity_encoder": "Linear + LayerNorm + GELU",
                "neighbor_interaction_encoder": "MLP over world-space neighbor/social features",
                "obstacle_feature_encoder": "MLP over nearest obstacle and boundary features",
                "temporal_module": "GRU over K-frame per-entity history",
                "stochastic_latent_intention": "4D sampled z injected into transition head during rollout",
                "transition_head": "MLP predicting residual acceleration in m/s^2",
                "collision_projection_layer": "post-transition world-space projection for disks and scene constraints",
            },
            "training": training_summary,
        },
        "synthetic_evaluation": evaluation["summary"],
        "synthetic_evaluation_meta": evaluation["meta"],
        "terminal_clusters": cluster_summary,
        "aerialmpt_real_data": aerial_summary,
        "conclusions": make_conclusions(evaluation["summary"], cluster_summary, aerial_summary),
    }

    (OUT_ROOT / "stage2_summary.json").write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    (REPORT_DIR / "report_stage2.md").write_text(build_stage2_report(report), encoding="utf-8")
    print(json.dumps({"summary": str(OUT_ROOT / "stage2_summary.json"), "report": str(REPORT_DIR / "report_stage2.md")}, indent=2), flush=True)
    print(json.dumps(key_console_summary(report), indent=2), flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_scene_templates() -> List[SceneSpec]:
    return [
        SceneSpec(
            name="bottleneck_corridor",
            width=42.0,
            height=26.0,
            obstacles=[
                Rect(18.5, 0.0, 22.0, 9.6, "wall"),
                Rect(18.5, 16.4, 22.0, 26.0, "wall"),
                Rect(9.0, 2.0, 13.0, 7.0, "booth"),
                Rect(29.0, 18.0, 34.5, 23.0, "booth"),
            ],
            exits={"west": (1.0, 13.0), "east": (41.0, 13.0), "north": (21.0, 1.0), "south": (21.0, 25.0)},
            spawn_regions=[
                {"rect": (1.0, 8.0, 7.0, 18.0), "goal": "east", "weight": 0.55},
                {"rect": (35.0, 8.0, 41.0, 18.0), "goal": "west", "weight": 0.30},
                {"rect": (16.0, 22.0, 26.0, 25.0), "goal": "north", "weight": 0.15},
            ],
            event_hint="congestion",
        ),
        SceneSpec(
            name="split_around_obstacle",
            width=42.0,
            height=26.0,
            obstacles=[
                Rect(17.0, 7.0, 25.0, 19.0, "large_booth"),
                Rect(6.0, 0.0, 9.0, 7.0, "wall"),
                Rect(6.0, 19.0, 9.0, 26.0, "wall"),
            ],
            exits={"west": (1.0, 13.0), "east": (41.0, 13.0), "north": (21.0, 1.0), "south": (21.0, 25.0)},
            spawn_regions=[
                {"rect": (1.0, 7.0, 6.0, 19.0), "goal": "east", "weight": 0.75},
                {"rect": (34.0, 4.0, 41.0, 10.0), "goal": "west", "weight": 0.15},
                {"rect": (34.0, 17.0, 41.0, 23.0), "goal": "west", "weight": 0.10},
            ],
            event_hint="detour",
        ),
        SceneSpec(
            name="merge_to_exit",
            width=42.0,
            height=26.0,
            obstacles=[
                Rect(12.0, 9.0, 30.0, 11.5, "barrier"),
                Rect(12.0, 14.5, 30.0, 17.0, "barrier"),
                Rect(31.5, 0.0, 34.0, 9.0, "wall"),
                Rect(31.5, 17.0, 34.0, 26.0, "wall"),
            ],
            exits={"west": (1.0, 13.0), "east": (41.0, 13.0), "north": (21.0, 1.0), "south": (21.0, 25.0)},
            spawn_regions=[
                {"rect": (2.0, 2.0, 12.0, 8.0), "goal": "east", "weight": 0.45},
                {"rect": (2.0, 18.0, 12.0, 24.0), "goal": "east", "weight": 0.45},
                {"rect": (35.0, 11.0, 41.0, 15.0), "goal": "west", "weight": 0.10},
            ],
            event_hint="merge",
        ),
        SceneSpec(
            name="crossing_flows",
            width=42.0,
            height=26.0,
            obstacles=[
                Rect(19.0, 11.0, 23.0, 15.0, "kiosk"),
                Rect(7.0, 7.0, 11.0, 11.0, "booth"),
                Rect(31.0, 15.0, 35.0, 19.0, "booth"),
            ],
            exits={"west": (1.0, 13.0), "east": (41.0, 13.0), "north": (21.0, 1.0), "south": (21.0, 25.0)},
            spawn_regions=[
                {"rect": (1.0, 9.0, 7.0, 17.0), "goal": "east", "weight": 0.38},
                {"rect": (35.0, 9.0, 41.0, 17.0), "goal": "west", "weight": 0.22},
                {"rect": (16.0, 20.0, 26.0, 25.0), "goal": "north", "weight": 0.25},
                {"rect": (16.0, 1.0, 26.0, 6.0), "goal": "south", "weight": 0.15},
            ],
            event_hint="split",
        ),
    ]


def generate_or_load_dataset(scenes: List[SceneSpec]) -> List[Dict]:
    manifest_path = DATASET_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        episodes = []
        for item in manifest["episodes"]:
            arrays = np.load(DATASET_DIR / item["npz"])
            scene = scene_from_dict(item["scene"])
            episodes.append({"meta": item, "scene": scene, "states": arrays["states"], "diagnostics": arrays["diagnostics"]})
        return episodes

    rng = np.random.default_rng(2026)
    split_counts = {"train": 24, "val": 6, "test": 8}
    episodes = []
    manifest = {"name": "SyntheticPhysicalCrowd2.5D", "dt": DT, "frames": EPISODE_FRAMES, "episodes": []}
    episode_id = 0
    for split, count in split_counts.items():
        for _ in range(count):
            scene = scenes[int(rng.integers(0, len(scenes)))]
            episode = simulate_episode(scene, episode_id, split, rng)
            filename = f"{split}_episode_{episode_id:04d}.npz"
            np.savez_compressed(DATASET_DIR / filename, states=episode["states"], diagnostics=episode["diagnostics"])
            meta = episode["meta"]
            meta["npz"] = filename
            meta["scene"] = scene_to_dict(scene)
            (DATASET_DIR / f"{split}_episode_{episode_id:04d}_scene.json").write_text(json.dumps(meta["scene"], indent=2), encoding="utf-8")
            manifest["episodes"].append(meta)
            episodes.append({"meta": meta, "scene": scene, "states": episode["states"], "diagnostics": episode["diagnostics"]})
            episode_id += 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return episodes


def simulate_episode(scene: SceneSpec, episode_id: int, split: str, rng: np.random.Generator) -> Dict:
    n_agents = int(rng.integers(10, 51))
    states = np.zeros((EPISODE_FRAMES, n_agents, len(STATE_COLUMNS)), dtype=np.float32)
    diagnostics = np.zeros((EPISODE_FRAMES, 8), dtype=np.float32)
    pause_timers = np.zeros(n_agents, dtype=np.int32)
    path_lengths = np.zeros(n_agents, dtype=np.float32)
    initial_positions = np.zeros((n_agents, 2), dtype=np.float32)
    switched_goal = np.zeros(n_agents, dtype=bool)

    for i in range(n_agents):
        x, y, gx, gy = sample_initial_agent(scene, states[0, :i], rng)
        radius = float(rng.uniform(0.24, 0.39))
        desired = float(rng.uniform(0.8, 1.55))
        states[0, i] = [x, y, 0.0, 0.0, 0.0, 0.0, radius, gx, gy, desired, 1.0]
        initial_positions[i] = [x, y]

    for t in range(EPISODE_FRAMES - 1):
        current = states[t].copy()
        for i in range(n_agents):
            if distance(current[i, :2], current[i, 7:9]) < 0.9:
                if rng.random() < 0.60:
                    current[i, 9] = 0.0
                    pause_timers[i] = max(pause_timers[i], int(rng.integers(5, 18)))
                elif rng.random() < 0.35:
                    gx, gy = random_exit_goal(scene, rng)
                    current[i, 7:9] = [gx, gy]
                    switched_goal[i] = True

            if pause_timers[i] > 0:
                pause_timers[i] -= 1
                current[i, 9] = min(current[i, 9], 0.1)
            elif rng.random() < 0.004:
                pause_timers[i] = int(rng.integers(4, 14))

            if rng.random() < 0.0025:
                gx, gy = random_exit_goal(scene, rng)
                current[i, 7:9] = [gx, gy]
                switched_goal[i] = True

        base_accel = compute_physics_acceleration(current, scene, include_scene=True)
        congestion = local_density_values(current)
        random_intent = rng.normal(0.0, 0.10, size=base_accel.shape)
        slowdown = np.clip(1.0 - 0.09 * congestion, 0.32, 1.0).reshape(-1, 1)
        accel = base_accel * slowdown + random_intent
        accel = clip_vectors(accel, MAX_ACCEL)
        next_state = integrate_state(current, accel, DT)
        projected, info = project_state_constraints(next_state, scene)
        path_lengths += np.linalg.norm(projected[:, :2] - current[:, :2], axis=1)
        projected[:, 4:6] = (projected[:, 2:4] - current[:, 2:4]) / DT
        states[t + 1] = projected
        diagnostics[t + 1] = diagnostics_row(projected, scene, info, path_lengths, initial_positions, switched_goal)

    event_label = classify_synthetic_episode(states, diagnostics, scene)
    meta = {
        "episode_id": int(episode_id),
        "split": split,
        "frames": EPISODE_FRAMES,
        "agents": int(n_agents),
        "scene_name": scene.name,
        "event_hint": scene.event_hint,
        "event_label": event_label,
        "state_columns": STATE_COLUMNS,
        "diagnostic_columns": [
            "min_gap_m",
            "attempted_collision_count",
            "collision_count_after_projection",
            "projection_cost_m",
            "obstacle_violation_count",
            "boundary_violation_m",
            "density_people_per_m2",
            "reached_fraction",
        ],
    }
    return {"meta": meta, "states": states, "diagnostics": diagnostics}


def sample_initial_agent(scene: SceneSpec, existing: np.ndarray, rng: np.random.Generator) -> Tuple[float, float, float, float]:
    weights = np.array([region["weight"] for region in scene.spawn_regions], dtype=float)
    weights /= weights.sum()
    for _ in range(600):
        region = scene.spawn_regions[int(rng.choice(len(scene.spawn_regions), p=weights))]
        x1, y1, x2, y2 = region["rect"]
        x = float(rng.uniform(x1, x2))
        y = float(rng.uniform(y1, y2))
        if point_in_any_rect((x, y), scene.obstacles):
            continue
        if len(existing):
            distances = np.linalg.norm(existing[:, :2] - np.array([x, y]), axis=1)
            if np.any(distances < existing[:, 6] + 0.42):
                continue
        goal = scene.exits[region["goal"]]
        gx = float(goal[0] + rng.normal(0.0, 0.45))
        gy = float(goal[1] + rng.normal(0.0, 0.45))
        return x, y, gx, gy
    goal = scene.exits["east"]
    return 2.0, scene.height / 2.0, goal[0], goal[1]


def random_exit_goal(scene: SceneSpec, rng: np.random.Generator) -> Tuple[float, float]:
    name = random.choice(list(scene.exits.keys()))
    goal = scene.exits[name]
    return float(goal[0] + rng.normal(0.0, 0.4)), float(goal[1] + rng.normal(0.0, 0.4))


def compute_physics_acceleration(state: np.ndarray, scene: SceneSpec, include_scene: bool = True) -> np.ndarray:
    n = state.shape[0]
    accel = np.zeros((n, 2), dtype=np.float32)
    positions = state[:, :2]
    velocities = state[:, 2:4]
    radii = state[:, 6]
    goals = state[:, 7:9]
    desired = state[:, 9]

    for i in range(n):
        direction = unit_vector_np(goals[i] - positions[i])
        desired_velocity = direction * desired[i]
        accel[i] += 1.10 * (desired_velocity - velocities[i])
        accel[i] += -0.10 * velocities[i]

        for j in range(n):
            if i == j:
                continue
            delta = positions[i] - positions[j]
            dist = max(1e-4, float(np.linalg.norm(delta)))
            normal = delta / dist
            gap = dist - float(radii[i] + radii[j])
            if gap < 2.2:
                rel_v = velocities[i] - velocities[j]
                closing = max(0.0, -float(np.dot(rel_v, normal)))
                strength = 0.58 * math.exp(-max(gap, -0.5) / 0.70) + 0.12 * closing
                accel[i] += normal * strength

        if include_scene:
            obstacle_vec, obstacle_dist, inside = nearest_obstacle_vector(tuple(positions[i]), scene)
            if inside:
                accel[i] += obstacle_vec * 2.2
            elif obstacle_dist < 1.9:
                accel[i] += obstacle_vec * (1.15 * (1.9 - obstacle_dist))

            bx, by = boundary_force(tuple(positions[i]), scene, margin=1.2)
            accel[i] += np.array([bx, by], dtype=np.float32)

    return clip_vectors(accel, MAX_ACCEL)


def integrate_state(state: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
    next_state = state.copy()
    next_state[:, 4:6] = accel
    next_state[:, 2:4] = clip_vectors(state[:, 2:4] + accel * dt, MAX_SPEED)
    next_state[:, :2] = state[:, :2] + next_state[:, 2:4] * dt
    next_state[:, 10] = 1.0
    return next_state


def project_state_constraints(state: np.ndarray, scene: SceneSpec) -> Tuple[np.ndarray, Dict]:
    projected = state.copy()
    attempted_collisions = 0
    projection_cost = 0.0
    max_penetration = 0.0

    for _ in range(3):
        moved = False
        for i in range(projected.shape[0]):
            for j in range(i + 1, projected.shape[0]):
                delta = projected[j, :2] - projected[i, :2]
                dist = max(1e-6, float(np.linalg.norm(delta)))
                min_dist = float(projected[i, 6] + projected[j, 6] + 0.03)
                penetration = min_dist - dist
                if penetration <= 0:
                    continue
                attempted_collisions += 1
                max_penetration = max(max_penetration, penetration)
                normal = delta / dist
                projected[i, :2] -= normal * penetration * 0.5
                projected[j, :2] += normal * penetration * 0.5
                projection_cost += penetration
                moved = True
        if not moved:
            break

    obstacle_violations = 0
    boundary_violation = 0.0
    for i in range(projected.shape[0]):
        r = float(projected[i, 6])
        old = projected[i, :2].copy()
        projected[i, 0] = float(np.clip(projected[i, 0], r, scene.width - r))
        projected[i, 1] = float(np.clip(projected[i, 1], r, scene.height - r))
        boundary_violation += float(np.linalg.norm(projected[i, :2] - old))

        for rect in scene.obstacles:
            if point_in_rect(tuple(projected[i, :2]), rect, pad=r):
                obstacle_violations += 1
                corrected = push_out_rect(tuple(projected[i, :2]), rect, pad=r + 0.04)
                projection_cost += distance_tuple(tuple(projected[i, :2]), corrected)
                projected[i, :2] = corrected

    info = {
        "attempted_collisions": attempted_collisions,
        "projection_cost": float(projection_cost),
        "max_penetration": float(max_penetration),
        "obstacle_violations": obstacle_violations,
        "boundary_violation": float(boundary_violation),
    }
    return projected, info


def diagnostics_row(
    state: np.ndarray,
    scene: SceneSpec,
    info: Dict,
    path_lengths: np.ndarray,
    initial_positions: np.ndarray,
    switched_goal: np.ndarray,
) -> np.ndarray:
    min_gap, collisions = min_gap_and_collisions(state)
    reached = np.linalg.norm(state[:, :2] - state[:, 7:9], axis=1) < 1.2
    density = density_people_per_m2_np(state)
    return np.array(
        [
            min_gap,
            info["attempted_collisions"],
            collisions,
            info["projection_cost"],
            info["obstacle_violations"],
            info["boundary_violation"],
            density,
            float(np.mean(reached)),
        ],
        dtype=np.float32,
    )


def classify_synthetic_episode(states: np.ndarray, diagnostics: np.ndarray, scene: SceneSpec) -> str:
    final_reached = float(diagnostics[-1, 7])
    mean_density = float(np.mean(diagnostics[:, 6]))
    low_speed_fraction = float(np.mean(np.linalg.norm(states[:, :, 2:4], axis=2) < 0.18))
    min_gap = float(np.min(diagnostics[:, 0]))
    final_positions = states[-1, :, :2]
    final_spread = float(np.std(final_positions[:, 0]) + np.std(final_positions[:, 1]))
    path_ratio = path_efficiency(states)

    if min_gap < -0.03:
        return "collision-risk"
    if scene.event_hint == "split" or final_spread > 12.0:
        return "split-flow"
    if path_ratio > 1.42 or scene.event_hint == "detour":
        return "detour"
    if low_speed_fraction > 0.34 and mean_density > 0.08:
        return "congestion"
    if final_reached > 0.70 and path_ratio < 1.35:
        return "smooth-pass"
    if low_speed_fraction > 0.45:
        return "stalled"
    if scene.event_hint in {"merge", "congestion"}:
        return "congestion"
    return scene.event_hint


def path_efficiency(states: np.ndarray) -> float:
    diffs = np.linalg.norm(np.diff(states[:, :, :2], axis=0), axis=2).sum(axis=0)
    straight = np.linalg.norm(states[-1, :, :2] - states[0, :, :2], axis=1)
    ratios = diffs / np.maximum(1.0, straight)
    return float(np.mean(ratios))


def build_training_tensors(dataset: List[Dict], split: str) -> Dict:
    entity_rows = []
    neighbor_rows = []
    obstacle_rows = []
    target_rows = []
    sample_meta = []
    episodes = [ep for ep in dataset if ep["meta"]["split"] == split]

    for episode in episodes:
        states = episode["states"]
        scene = episode["scene"]
        for t in range(K_HISTORY - 1, EPISODE_FRAMES - 1):
            current = states[t]
            base_accel = compute_physics_acceleration(current, scene, include_scene=True)
            target_residual = states[t + 1, :, 4:6] - base_accel
            for i in range(current.shape[0]):
                entity_rows.append(entity_sequence_features(states[t - K_HISTORY + 1 : t + 1], scene, i))
                neighbor_rows.append(neighbor_features(current, i))
                obstacle_rows.append(obstacle_features(current, scene, i))
                target_rows.append(target_residual[i])
                sample_meta.append((episode["meta"]["episode_id"], t, i))

    entity = np.asarray(entity_rows, dtype=np.float32)
    neighbor = np.asarray(neighbor_rows, dtype=np.float32)
    obstacle = np.asarray(obstacle_rows, dtype=np.float32)
    target = np.asarray(target_rows, dtype=np.float32)

    if split == "train":
        normalization = {
            "entity_mean": entity.mean(axis=(0, 1)).tolist(),
            "entity_std": (entity.std(axis=(0, 1)) + 1e-6).tolist(),
            "neighbor_mean": neighbor.mean(axis=0).tolist(),
            "neighbor_std": (neighbor.std(axis=0) + 1e-6).tolist(),
            "obstacle_mean": obstacle.mean(axis=0).tolist(),
            "obstacle_std": (obstacle.std(axis=0) + 1e-6).tolist(),
            "target_mean": target.mean(axis=0).tolist(),
            "target_std": (target.std(axis=0) + 1e-6).tolist(),
        }
    else:
        train_norm_path = OUT_ROOT / "_normalization_tmp.json"
        normalization = json.loads(train_norm_path.read_text()) if train_norm_path.exists() else None
        if normalization is None:
            normalization = {
                "entity_mean": [0.0] * entity.shape[-1],
                "entity_std": [1.0] * entity.shape[-1],
                "neighbor_mean": [0.0] * neighbor.shape[-1],
                "neighbor_std": [1.0] * neighbor.shape[-1],
                "obstacle_mean": [0.0] * obstacle.shape[-1],
                "obstacle_std": [1.0] * obstacle.shape[-1],
                "target_mean": [0.0, 0.0],
                "target_std": [1.0, 1.0],
            }

    if split == "train":
        (OUT_ROOT / "_normalization_tmp.json").write_text(json.dumps(normalization), encoding="utf-8")

    entity_n = normalize_array(entity, normalization["entity_mean"], normalization["entity_std"])
    neighbor_n = normalize_array(neighbor, normalization["neighbor_mean"], normalization["neighbor_std"])
    obstacle_n = normalize_array(obstacle, normalization["obstacle_mean"], normalization["obstacle_std"])
    target_n = normalize_array(target, normalization["target_mean"], normalization["target_std"])

    return {
        "entity": entity_n,
        "neighbor": neighbor_n,
        "obstacle": obstacle_n,
        "target": target_n,
        "target_raw": target,
        "normalization": normalization,
        "meta": sample_meta,
    }


def entity_sequence_features(history: np.ndarray, scene: SceneSpec, agent_index: int) -> np.ndarray:
    features = []
    for frame in history:
        agent = frame[agent_index]
        pos = agent[:2]
        vel = agent[2:4]
        acc = agent[4:6]
        goal_vec = agent[7:9] - pos
        goal_dir = unit_vector_np(goal_vec)
        obstacle_vec, obstacle_dist, inside = nearest_obstacle_vector(tuple(pos), scene)
        boundary_dist = min(pos[0], scene.width - pos[0], pos[1], scene.height - pos[1])
        density = local_density_for_agent(frame, agent_index)
        features.append(
            [
                pos[0] / scene.width,
                pos[1] / scene.height,
                vel[0] / MAX_SPEED,
                vel[1] / MAX_SPEED,
                acc[0] / MAX_ACCEL,
                acc[1] / MAX_ACCEL,
                agent[6],
                goal_dir[0],
                goal_dir[1],
                agent[9] / MAX_SPEED,
                min(obstacle_dist, 5.0) / 5.0,
                min(boundary_dist, 5.0) / 5.0,
                density / 10.0,
                1.0 if inside else 0.0,
            ]
        )
    return np.asarray(features, dtype=np.float32)


def neighbor_features(state: np.ndarray, agent_index: int) -> np.ndarray:
    agent = state[agent_index]
    positions = state[:, :2]
    velocities = state[:, 2:4]
    delta = positions - agent[:2]
    distances = np.linalg.norm(delta, axis=1)
    distances[agent_index] = np.inf
    order = np.argsort(distances)[:6]
    valid = order[np.isfinite(distances[order])]
    if len(valid) == 0:
        return np.zeros(12, dtype=np.float32)

    rel = delta[valid]
    rel_v = velocities[valid] - agent[2:4]
    d = distances[valid]
    gaps = d - (state[valid, 6] + agent[6])
    nearest = valid[0]
    nearest_delta = delta[nearest]
    nearest_d = max(1e-6, distances[nearest])
    bearing = math.atan2(nearest_delta[1], nearest_delta[0])
    ttc = time_to_collision(agent[:2], agent[2:4], state[nearest, :2], state[nearest, 2:4], agent[6] + state[nearest, 6])
    density = float(np.sum(distances < 2.0))
    return np.array(
        [
            float(np.mean(rel[:, 0]) / 5.0),
            float(np.mean(rel[:, 1]) / 5.0),
            float(np.mean(d) / 5.0),
            float(np.min(d) / 5.0),
            float(np.mean(rel_v[:, 0]) / MAX_SPEED),
            float(np.mean(rel_v[:, 1]) / MAX_SPEED),
            float(np.min(gaps)),
            float(np.mean(gaps)),
            math.sin(bearing),
            math.cos(bearing),
            min(ttc, 8.0) / 8.0,
            density / 10.0,
        ],
        dtype=np.float32,
    )


def obstacle_features(state: np.ndarray, scene: SceneSpec, agent_index: int) -> np.ndarray:
    agent = state[agent_index]
    pos = tuple(agent[:2])
    obstacle_vec, obstacle_dist, inside = nearest_obstacle_vector(pos, scene)
    left, right, bottom, top = agent[0], scene.width - agent[0], agent[1], scene.height - agent[1]
    boundary_dist = min(left, right, bottom, top)
    boundary_dir = np.array([0.0, 0.0], dtype=np.float32)
    if boundary_dist == left:
        boundary_dir = np.array([1.0, 0.0])
    elif boundary_dist == right:
        boundary_dir = np.array([-1.0, 0.0])
    elif boundary_dist == bottom:
        boundary_dir = np.array([0.0, 1.0])
    else:
        boundary_dir = np.array([0.0, -1.0])
    return np.array(
        [
            obstacle_vec[0],
            obstacle_vec[1],
            min(obstacle_dist, 6.0) / 6.0,
            1.0 if inside else 0.0,
            boundary_dir[0],
            boundary_dir[1],
            min(boundary_dist, 6.0) / 6.0,
            obstacle_count_near(pos, scene, 3.0) / 4.0,
        ],
        dtype=np.float32,
    )


def train_residual_model(train_pack: Dict, val_pack: Dict) -> Tuple[ResidualCrowdModel, Dict]:
    model = ResidualCrowdModel(
        entity_dim=train_pack["entity"].shape[-1],
        neighbor_dim=train_pack["neighbor"].shape[-1],
        obstacle_dim=train_pack["obstacle"].shape[-1],
    ).to(DEVICE)
    dataset = TensorDataset(
        torch.tensor(train_pack["entity"], dtype=torch.float32),
        torch.tensor(train_pack["neighbor"], dtype=torch.float32),
        torch.tensor(train_pack["obstacle"], dtype=torch.float32),
        torch.tensor(train_pack["target"], dtype=torch.float32),
    )
    val_tensors = (
        torch.tensor(val_pack["entity"], dtype=torch.float32),
        torch.tensor(val_pack["neighbor"], dtype=torch.float32),
        torch.tensor(val_pack["obstacle"], dtype=torch.float32),
        torch.tensor(val_pack["target"], dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=768, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0018, weight_decay=0.0005)
    loss_fn = nn.SmoothL1Loss()
    best_state = None
    best_val = float("inf")
    history = []

    for epoch in range(1, 9):
        model.train()
        losses = []
        for entity, neighbor, obstacle, target in loader:
            z = torch.randn((entity.shape[0], model.latent_dim), dtype=torch.float32) * 0.35
            pred = model(entity, neighbor, obstacle, z)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            entity, neighbor, obstacle, target = val_tensors
            chunks = []
            for start in range(0, entity.shape[0], 4096):
                z = torch.zeros((min(4096, entity.shape[0] - start), model.latent_dim), dtype=torch.float32)
                chunks.append(model(entity[start : start + 4096], neighbor[start : start + 4096], obstacle[start : start + 4096], z))
            pred = torch.cat(chunks, dim=0)
            val_loss = float(loss_fn(pred, target).item())
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_smooth_l1": round(train_loss, 5), "val_smooth_l1": round(val_loss, 5)})
        print(f"epoch={epoch:02d} train={train_loss:.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "train_samples": int(train_pack["entity"].shape[0]),
        "val_samples": int(val_pack["entity"].shape[0]),
        "epochs": len(history),
        "best_val_smooth_l1_scaled": round(best_val, 5),
        "history": history,
        "target_residual_std_mps2": [round(float(x), 4) for x in np.std(train_pack["target_raw"], axis=0)],
    }


def evaluate_smc_proposals(dataset: List[Dict], model: ResidualCrowdModel, normalization: Dict) -> Dict:
    all_test_episodes = [ep for ep in dataset if ep["meta"]["split"] == "test"]
    test_episodes = sorted(all_test_episodes, key=lambda ep: ep["meta"]["agents"])[:EVAL_TEST_EPISODES]
    methods = ["physics", "neural", "hybrid"]
    all_results = {method: [] for method in methods}
    for episode in test_episodes:
        print(f"evaluating episode={episode['meta']['episode_id']} agents={episode['meta']['agents']}", flush=True)
        for method in methods:
            print(f"  proposal={method}", flush=True)
            rollout = run_smc_rollout_episode(episode, model, normalization, method)
            all_results[method].append(rollout)

    summary = {}
    for method in methods:
        summary[method] = aggregate_method_metrics(all_results[method])
    return {
        "rollouts": all_results,
        "summary": summary,
        "meta": {
            "full_test_episodes": len(all_test_episodes),
            "evaluated_test_episodes": len(test_episodes),
            "evaluated_episode_ids": [int(ep["meta"]["episode_id"]) for ep in test_episodes],
            "reason": "64-particle t+100 SMC is expensive; this run evaluates a clearly marked smallest-agent test subset.",
        },
    }


def run_smc_rollout_episode(episode: Dict, model: ResidualCrowdModel, normalization: Dict, method: str) -> Dict:
    rng = np.random.default_rng(10_000 + episode["meta"]["episode_id"] * 17 + ["physics", "neural", "hybrid"].index(method))
    states = episode["states"]
    scene = episode["scene"]
    start_t = K_HISTORY
    horizon = 100
    initial_history = states[start_t - K_HISTORY + 1 : start_t + 1]
    histories = np.repeat(initial_history[None, :, :, :], PARTICLES, axis=0).astype(np.float32)
    histories[:, -1, :, :2] += rng.normal(0, 0.04, size=histories[:, -1, :, :2].shape)
    trajectories = np.zeros((PARTICLES, horizon + 1, states.shape[1], states.shape[2]), dtype=np.float32)
    log_weights = np.zeros(PARTICLES, dtype=np.float64)
    weights = np.ones(PARTICLES, dtype=np.float64) / PARTICLES
    diagnostics = []
    trajectories[:, 0] = histories[:, -1]

    for step in range(1, horizon + 1):
        accel_batch, proposal_logp = proposal_acceleration_batch(histories, scene, model, normalization, method, rng)
        for p in range(PARTICLES):
            current = histories[p, -1]
            accel = accel_batch[p]
            next_state = integrate_state(current, accel, DT)
            projected, info = project_state_constraints(next_state, scene)
            projected[:, 4:6] = (projected[:, 2:4] - current[:, 2:4]) / DT
            log_weights[p] += proposal_logp[p]
            log_weights[p] -= 3.8 * info["projection_cost"] + 1.5 * info["obstacle_violations"] + 1.0 * info["boundary_violation"]
            log_weights[p] -= speed_accel_penalty(projected)
            histories[p] = np.concatenate([histories[p, 1:], projected[None, :, :]], axis=0)
            trajectories[p, step] = projected
        weights = normalize_log_weights(log_weights)
        if effective_sample_size(weights) < PARTICLES * 0.45 and step < horizon:
            indexes = systematic_resample_indexes(weights, rng)
            histories = histories[indexes]
            trajectories = trajectories[indexes]
            log_weights = np.zeros(PARTICLES, dtype=np.float64)
            weights = np.ones(PARTICLES, dtype=np.float64) / PARTICLES
        if step in HORIZONS:
            diagnostics.append({"step": step, "ESS": round(float(effective_sample_size(weights)), 3)})

    true_future = states[start_t : start_t + horizon + 1]
    metrics = rollout_metrics(trajectories, weights, true_future, scene, episode["meta"]["event_label"])
    clusters = semantic_terminal_clusters(trajectories, weights, true_future, scene, episode["meta"]["event_label"])
    return {
        "episode_id": episode["meta"]["episode_id"],
        "scene": scene.name,
        "true_event": episode["meta"]["event_label"],
        "method": method,
        "weights": weights,
        "metrics": metrics,
        "clusters": clusters,
        "diagnostics": diagnostics,
    }


def proposal_acceleration_batch(
    histories: np.ndarray,
    scene: SceneSpec,
    model: ResidualCrowdModel,
    normalization: Dict,
    method: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    particle_count, _, n_agents, _ = histories.shape
    base = np.zeros((particle_count, n_agents, 2), dtype=np.float32)
    for p in range(particle_count):
        base[p] = compute_physics_acceleration(histories[p, -1], scene, include_scene=True)

    if method == "physics":
        noise_std = 0.11
        noise = rng.normal(0, noise_std, size=base.shape).astype(np.float32)
        logp = -0.5 * np.mean((noise / noise_std) ** 2, axis=(1, 2))
        return np.stack([clip_vectors(base[p] + noise[p], MAX_ACCEL) for p in range(particle_count)]), logp

    residual = predict_residual_acceleration_batch(histories, scene, model, normalization, rng)
    if method == "neural":
        accel = residual
        noise_std = 0.16
    else:
        accel = base + residual
        noise_std = 0.09
    noise = rng.normal(0, noise_std, size=accel.shape).astype(np.float32)
    logp = -0.5 * np.mean((noise / noise_std) ** 2, axis=(1, 2))
    return np.stack([clip_vectors(accel[p] + noise[p], MAX_ACCEL) for p in range(particle_count)]), logp


def predict_residual_acceleration_batch(
    histories: np.ndarray,
    scene: SceneSpec,
    model: ResidualCrowdModel,
    normalization: Dict,
    rng: np.random.Generator,
) -> np.ndarray:
    particle_count, _, n_agents, _ = histories.shape
    entity = []
    neighbor = []
    obstacle = []
    for p in range(particle_count):
        current = histories[p, -1]
        for i in range(n_agents):
            entity.append(entity_sequence_features(histories[p], scene, i))
            neighbor.append(neighbor_features(current, i))
            obstacle.append(obstacle_features(current, scene, i))
    entity_n = normalize_array(np.asarray(entity, dtype=np.float32), normalization["entity_mean"], normalization["entity_std"])
    neighbor_n = normalize_array(np.asarray(neighbor, dtype=np.float32), normalization["neighbor_mean"], normalization["neighbor_std"])
    obstacle_n = normalize_array(np.asarray(obstacle, dtype=np.float32), normalization["obstacle_mean"], normalization["obstacle_std"])
    preds = []
    with torch.no_grad():
        for start in range(0, entity_n.shape[0], 4096):
            end = min(entity_n.shape[0], start + 4096)
            z = torch.tensor(rng.normal(0.0, 0.65, size=(end - start, model.latent_dim)), dtype=torch.float32)
            pred = model(
                torch.tensor(entity_n[start:end], dtype=torch.float32),
                torch.tensor(neighbor_n[start:end], dtype=torch.float32),
                torch.tensor(obstacle_n[start:end], dtype=torch.float32),
                z,
            ).numpy()
            preds.append(pred)
    residual = denormalize_array(np.concatenate(preds, axis=0), normalization["target_mean"], normalization["target_std"])
    return residual.reshape(particle_count, n_agents, 2).astype(np.float32)


def proposal_acceleration(
    current: np.ndarray,
    history: np.ndarray,
    scene: SceneSpec,
    model: ResidualCrowdModel,
    normalization: Dict,
    method: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    base = compute_physics_acceleration(current, scene, include_scene=True)
    if method == "physics":
        noise_std = 0.11
        noise = rng.normal(0, noise_std, size=base.shape)
        return clip_vectors(base + noise, MAX_ACCEL), float(-0.5 * np.mean((noise / noise_std) ** 2))

    residual = predict_residual_acceleration(history, scene, model, normalization, rng)
    if method == "neural":
        accel = residual
        noise_std = 0.16
    else:
        accel = base + residual
        noise_std = 0.09
    noise = rng.normal(0, noise_std, size=accel.shape)
    return clip_vectors(accel + noise, MAX_ACCEL), float(-0.5 * np.mean((noise / noise_std) ** 2))


def predict_residual_acceleration(
    history: np.ndarray,
    scene: SceneSpec,
    model: ResidualCrowdModel,
    normalization: Dict,
    rng: np.random.Generator,
) -> np.ndarray:
    entity = []
    neighbor = []
    obstacle = []
    current = history[-1]
    for i in range(current.shape[0]):
        entity.append(entity_sequence_features(history, scene, i))
        neighbor.append(neighbor_features(current, i))
        obstacle.append(obstacle_features(current, scene, i))
    entity_n = normalize_array(np.asarray(entity, dtype=np.float32), normalization["entity_mean"], normalization["entity_std"])
    neighbor_n = normalize_array(np.asarray(neighbor, dtype=np.float32), normalization["neighbor_mean"], normalization["neighbor_std"])
    obstacle_n = normalize_array(np.asarray(obstacle, dtype=np.float32), normalization["obstacle_mean"], normalization["obstacle_std"])
    with torch.no_grad():
        z = torch.tensor(rng.normal(0.0, 0.65, size=(current.shape[0], model.latent_dim)), dtype=torch.float32)
        pred = model(
            torch.tensor(entity_n, dtype=torch.float32),
            torch.tensor(neighbor_n, dtype=torch.float32),
            torch.tensor(obstacle_n, dtype=torch.float32),
            z,
        ).numpy()
    residual = denormalize_array(pred, normalization["target_mean"], normalization["target_std"])
    return residual.astype(np.float32)


def rollout_metrics(trajectories: np.ndarray, weights: np.ndarray, true_future: np.ndarray, scene: SceneSpec, true_event: str) -> Dict:
    weighted = np.tensordot(weights, trajectories, axes=(0, 0))
    out = {"horizons": {}}
    for h in HORIZONS:
        pred = weighted[: h + 1, :, :2]
        truth = true_future[: h + 1, :, :2]
        errors = np.linalg.norm(pred[1:] - truth[1:], axis=2)
        fde = np.linalg.norm(pred[h] - truth[h], axis=1)
        particle_ade = np.mean(np.linalg.norm(trajectories[:, 1 : h + 1, :, :2] - truth[None, 1 : h + 1, :, :2], axis=3), axis=(1, 2))
        particle_fde = np.mean(np.linalg.norm(trajectories[:, h, :, :2] - truth[None, h, :, :2], axis=2), axis=1)
        out["horizons"][str(h)] = {
            "ADE_m": round(float(np.mean(errors)), 3),
            "FDE_m": round(float(np.mean(fde)), 3),
            "best_of_64_ADE_m": round(float(np.min(particle_ade)), 3),
            "best_of_64_FDE_m": round(float(np.min(particle_fde)), 3),
        }
    out["coverage_t100_fde_lt_2m"] = bool(np.min(np.mean(np.linalg.norm(trajectories[:, 100, :, :2] - true_future[100, :, :2], axis=2), axis=1)) < 2.0)
    out.update(physical_violation_metrics(trajectories, scene))
    out["endpoint_nll_t100"] = round(endpoint_particle_nll(trajectories[:, 100, :, :2], weights, true_future[100, :, :2]), 3)
    out["cluster_diversity_score"] = None
    return out


def physical_violation_metrics(trajectories: np.ndarray, scene: SceneSpec) -> Dict:
    total_frames = trajectories.shape[0] * trajectories.shape[1]
    collision_frames = 0
    obstacle_frames = 0
    boundary_frames = 0
    speed_violations = 0
    accel_violations = 0
    jerk_values = []
    for particle in trajectories:
        accel = particle[:, :, 4:6]
        jerk_values.append(np.linalg.norm(np.diff(accel, axis=0), axis=2).mean())
        for frame in particle:
            min_gap, collisions = min_gap_and_collisions(frame)
            if collisions > 0 or min_gap < -1e-4:
                collision_frames += 1
            obstacle = 0
            boundary = 0
            for agent in frame:
                if agent[0] < agent[6] or agent[0] > scene.width - agent[6] or agent[1] < agent[6] or agent[1] > scene.height - agent[6]:
                    boundary += 1
                if point_in_any_rect(tuple(agent[:2]), scene.obstacles, pad=float(agent[6])):
                    obstacle += 1
                if np.linalg.norm(agent[2:4]) > MAX_SPEED + 1e-4:
                    speed_violations += 1
                if np.linalg.norm(agent[4:6]) > MAX_ACCEL + 1e-4:
                    accel_violations += 1
            if obstacle:
                obstacle_frames += 1
            if boundary:
                boundary_frames += 1
    agents_total = trajectories.shape[0] * trajectories.shape[1] * trajectories.shape[2]
    return {
        "collision_violation_rate": round(collision_frames / max(1, total_frames), 5),
        "obstacle_violation_rate": round(obstacle_frames / max(1, total_frames), 5),
        "boundary_violation_rate": round(boundary_frames / max(1, total_frames), 5),
        "max_speed_violation_rate": round(speed_violations / max(1, agents_total), 5),
        "acceleration_violation_rate": round(accel_violations / max(1, agents_total), 5),
        "trajectory_smoothness_mean_jerk": round(float(np.mean(jerk_values)), 3),
    }


def endpoint_particle_nll(particle_endpoints: np.ndarray, weights: np.ndarray, truth: np.ndarray, sigma: float = 1.25) -> float:
    log_probs = []
    norm_const = -math.log(2 * math.pi * sigma * sigma)
    for i in range(truth.shape[0]):
        d2 = np.sum((particle_endpoints[:, i, :] - truth[i]) ** 2, axis=1)
        components = np.log(np.maximum(weights, 1e-12)) + norm_const - 0.5 * d2 / (sigma * sigma)
        m = np.max(components)
        log_probs.append(m + math.log(float(np.sum(np.exp(components - m)))))
    return float(-np.mean(log_probs))


def semantic_terminal_clusters(
    trajectories: np.ndarray,
    weights: np.ndarray,
    true_future: np.ndarray,
    scene: SceneSpec,
    true_event: str,
) -> List[Dict]:
    particle_features = []
    labels = []
    for particle in trajectories:
        features = semantic_event_features(particle, scene)
        particle_features.append(features)
        labels.append(semantic_label_from_features(features))
    unique_labels = sorted(set(labels))
    clusters = []
    for label in unique_labels:
        indexes = [i for i, item in enumerate(labels) if item == label]
        mass = float(np.sum(weights[indexes]))
        if not indexes:
            continue
        representative = indexes[int(np.argmax(weights[indexes]))]
        ade_values = []
        fde_values = []
        violations = []
        for index in indexes:
            err = np.linalg.norm(trajectories[index, 1:101, :, :2] - true_future[1:101, :, :2], axis=2)
            ade_values.append(float(np.mean(err)))
            fde_values.append(float(np.mean(np.linalg.norm(trajectories[index, 100, :, :2] - true_future[100, :, :2], axis=1))))
            violations.append(physical_score(trajectories[index], scene))
        clusters.append(
            {
                "semantic_label": label,
                "probability": round(mass, 5),
                "representative_particle": int(representative),
                "average_ADE_m": round(float(np.mean(ade_values)), 3),
                "average_FDE_m": round(float(np.mean(fde_values)), 3),
                "average_physical_violation_rate": round(float(np.mean(violations)), 5),
                "credible": bool(mass > 0.08 and np.mean(violations) < 0.03),
                "naming_reason": naming_reason(label),
                "matches_true_event": bool(label == true_event),
            }
        )
    return sorted(clusters, key=lambda x: x["probability"], reverse=True)


def semantic_event_features(trajectory: np.ndarray, scene: SceneSpec) -> Dict:
    final = trajectory[-1]
    reached = np.linalg.norm(final[:, :2] - final[:, 7:9], axis=1) < 1.4
    speeds = np.linalg.norm(trajectory[:, :, 2:4], axis=2)
    low_speed = float(np.mean(speeds < 0.22))
    densities = [density_people_per_m2_np(frame) for frame in trajectory]
    high_density = float(np.mean(np.array(densities) > 0.12))
    min_gaps = [min_gap_and_collisions(frame)[0] for frame in trajectory]
    collision_risk = float(np.min(min_gaps) < 0.12)
    path_ratio = path_efficiency(trajectory)
    final_spread = float(np.std(final[:, 0]) + np.std(final[:, 1]))
    split = final_spread > 11.0
    physical = physical_score(trajectory, scene)
    pass_time = first_reach_time(trajectory)
    return {
        "reached_exit_fraction": float(np.mean(reached)),
        "jammed_fraction": low_speed,
        "detour_score": path_ratio,
        "high_density_fraction": high_density,
        "collision_risk": collision_risk,
        "average_pass_time": pass_time,
        "final_spread": final_spread,
        "split_flow": float(split),
        "stalled_fraction": low_speed,
        "physical_violation_rate": physical,
    }


def semantic_label_from_features(features: Dict) -> str:
    if features["collision_risk"] > 0.5:
        return "collision-risk"
    if features["stalled_fraction"] > 0.42:
        return "stalled"
    if features["jammed_fraction"] > 0.28 and features["high_density_fraction"] > 0.15:
        return "congestion"
    if features["split_flow"] > 0.5:
        return "split-flow"
    if features["detour_score"] > 1.45:
        return "detour"
    if features["reached_exit_fraction"] > 0.65:
        return "smooth-pass"
    return "partial-flow"


def build_terminal_cluster_report(evaluation: Dict) -> Dict:
    report = {}
    for method, rollouts in evaluation["rollouts"].items():
        clusters_by_label: Dict[str, List[Dict]] = {}
        for rollout in rollouts:
            for cluster in rollout["clusters"]:
                clusters_by_label.setdefault(cluster["semantic_label"], []).append(cluster)
        rows = []
        for label, clusters in clusters_by_label.items():
            probability = float(np.mean([cluster["probability"] for cluster in clusters]))
            rows.append(
                {
                    "semantic_label": label,
                    "mean_probability": round(probability, 5),
                    "episodes_present": len(clusters),
                    "mean_ADE_m": round(float(np.mean([cluster["average_ADE_m"] for cluster in clusters])), 3),
                    "mean_FDE_m": round(float(np.mean([cluster["average_FDE_m"] for cluster in clusters])), 3),
                    "mean_physical_violation_rate": round(float(np.mean([cluster["average_physical_violation_rate"] for cluster in clusters])), 5),
                    "credible_fraction": round(float(np.mean([cluster["credible"] for cluster in clusters])), 3),
                    "naming_reason": naming_reason(label),
                }
            )
        total_prob = sum(row["mean_probability"] for row in rows)
        probs = np.array([row["mean_probability"] / max(1e-12, total_prob) for row in rows], dtype=float)
        diversity = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum() / max(1e-12, math.log(max(2, len(rows)))))
        report[method] = {"cluster_diversity_score": round(diversity, 3), "clusters": sorted(rows, key=lambda x: x["mean_probability"], reverse=True)}
        evaluation["summary"][method]["cluster_diversity_score"] = round(diversity, 3)
    return report


def aggregate_method_metrics(results: List[Dict]) -> Dict:
    summary = {"horizons": {}}
    for h in HORIZONS:
        for key in ["ADE_m", "FDE_m", "best_of_64_ADE_m", "best_of_64_FDE_m"]:
            summary["horizons"].setdefault(str(h), {})[key] = round(float(np.mean([r["metrics"]["horizons"][str(h)][key] for r in results])), 3)
    scalar_keys = [
        "coverage_t100_fde_lt_2m",
        "collision_violation_rate",
        "obstacle_violation_rate",
        "boundary_violation_rate",
        "max_speed_violation_rate",
        "acceleration_violation_rate",
        "trajectory_smoothness_mean_jerk",
        "endpoint_nll_t100",
    ]
    for key in scalar_keys:
        summary[key] = round(float(np.mean([r["metrics"][key] for r in results])), 5)
    top_correct = []
    for result in results:
        top = result["clusters"][0]["semantic_label"] if result["clusters"] else None
        top_correct.append(top == result["true_event"])
    summary["semantic_event_accuracy_top_cluster"] = round(float(np.mean(top_correct)), 3)
    return summary


def load_aerialmpt_stage1_summary() -> Dict:
    path = ROOT / "experiments" / "outputs" / "pseudo3d_world_model" / "summary.json"
    if not path.exists():
        return {"available": False, "reason": "stage-1 pseudo3d summary missing"}
    summary = json.loads(path.read_text())
    rows = summary.get("horizon_diagnostics", [])
    free_t12 = [row for row in rows if row.get("mode") == "free-run world rollout" and row.get("horizon") == 12]
    return {
        "available": True,
        "dataset": summary.get("dataset", {}),
        "calibration_quality": summary.get("calibration_quality", {}),
        "t12_free_run_metrics": free_t12[0] if free_t12 else None,
        "t100_status": "free-run only; no ground truth in selected AerialMPT sequence",
        "cluster_limitation": "stage-1 AerialMPT terminal clusters were all jammed/east-shifted, so semantic diversity was insufficient.",
    }


def make_conclusions(synthetic_summary: Dict, clusters: Dict, aerial: Dict) -> Dict:
    physics = synthetic_summary["physics"]
    neural = synthetic_summary["neural"]
    hybrid = synthetic_summary["hybrid"]
    h100 = "100"
    hybrid_beats_physics = hybrid["horizons"][h100]["ADE_m"] < physics["horizons"][h100]["ADE_m"]
    hybrid_beats_neural = hybrid["horizons"][h100]["ADE_m"] < neural["horizons"][h100]["ADE_m"]
    return {
        "is_now_learned_world_model": True,
        "synthetic_t100_reliability": "measurable on synthetic test; reliability depends on the reported ADE/FDE, coverage, and event accuracy rather than assumption.",
        "real_t12_reliability": "limited but measurable on AerialMPT bauma3 through stage-1 weak homography metrics.",
        "real_t100_status": "trend-only free-run; not accuracy-evaluable on bauma3.",
        "learned_residual_vs_physics": "hybrid residual improves t+100 ADE over hand-coded physics" if hybrid_beats_physics else "hybrid residual did not beat hand-coded physics on t+100 ADE",
        "neural_vs_hybrid": "hybrid beats neural-only" if hybrid_beats_neural else "neural-only matched or beat hybrid on t+100 ADE",
        "smc_coverage": {
            "physics": physics["coverage_t100_fde_lt_2m"],
            "neural": neural["coverage_t100_fde_lt_2m"],
            "hybrid": hybrid["coverage_t100_fde_lt_2m"],
        },
        "physical_constraints_effect": "collision/wall metrics should be read directly; projection is active in all three proposal types.",
        "terminal_cluster_diversity": {method: clusters[method]["cluster_diversity_score"] for method in clusters},
        "largest_failure_case": infer_largest_failure(synthetic_summary),
        "next_real_data_needed": [
            "real long trajectories with at least 100 future frames",
            "camera calibration or ground-plane control points",
            "scene walkability and obstacle polygons",
            "goal/exit annotations or destination labels",
            "body footprint calibration for pedestrian radius",
        ],
    }


def infer_largest_failure(summary: Dict) -> str:
    candidates = []
    for method, metrics in summary.items():
        candidates.append((metrics["horizons"]["100"]["FDE_m"], method))
    worst = max(candidates)[1]
    return f"{worst} proposal has the largest t+100 FDE among compared methods."


def build_stage2_report(report: Dict) -> str:
    synthetic = report["synthetic_evaluation"]
    synthetic_meta = report.get("synthetic_evaluation_meta", {})
    model = report["model"]
    clusters = report["terminal_clusters"]
    aerial = report["aerialmpt_real_data"]
    conclusions = report["conclusions"]

    return f"""# Stage 2 Learned 2.5D Crowd World Model

## 1. Audit Result

See `outputs/reports/model_audit.md`.

The previous pseudo-3D system was a state-space world-model scaffold, not a full learned world model. It had weak homography, hand-written physics, SMC rollout, and world-space collision projection, but no learned world-coordinate transition model.

The AerialMPT `bauma3` sequence still has no true `t+100` label. SyntheticPhysicalCrowd2.5D is introduced specifically to make `t+100` measurable.

## 2. SyntheticPhysicalCrowd2.5D

{markdown_table([report['dataset']])}

Environment properties:

- Episodes have {EPISODE_FRAMES} frames.
- Each episode has 10 to 50 pedestrians.
- Scenes include obstacles, walls, narrow passages, and exits.
- Each pedestrian has true goal, radius, velocity, acceleration, and desired speed.
- Projection enforces non-overlap and wall/obstacle constraints.
- Random intention changes and short stops create split, merge, congestion, detour, and stalled modes.
- Train / val / test are split by episode, not by frame.

## 3. Learned Transition Model

Architecture:

{markdown_table([model['architecture']])}

Training:

{markdown_table([model['training']])}

The neural component predicts residual acceleration in world coordinates. Final hybrid rollout is:

```text
A_final = A_hand_physics + A_neural_residual + epsilon
state_{{t+1}} = project_constraints(integrate(state_t, A_final))
```

## 4. SMC Proposal Comparison

Evaluation subset:

{markdown_table([synthetic_meta])}

### Horizon Metrics

{horizon_comparison_table(synthetic)}

### Physical Consistency / Calibration

{proposal_scalar_table(synthetic)}

Interpretation should be conservative: lower ADE/FDE is better, but not at the cost of collision, wall, speed, or acceleration violations.

## 5. Semantic Terminal Clustering

The terminal clustering now uses semantic event features:

- reached exit,
- jammed / low-speed time,
- detour score,
- high-density time,
- collision risk,
- average pass time,
- final spatial spread,
- split-flow indicator,
- stalled fraction,
- physical violation score.

{terminal_cluster_tables(clusters)}

## 6. Real AerialMPT Re-Application

Synthetic data has true `t+100`. AerialMPT bauma3 does not.

{markdown_table([aerial.get('dataset', {})])}

Real-data t+12 diagnostic:

{markdown_table([aerial.get('t12_free_run_metrics', {})] if aerial.get('t12_free_run_metrics') else [])}

Real-data t+100 status:

```text
{aerial.get('t100_status')}
```

The learned synthetic model is not presented as accurate on AerialMPT because domain transfer would require calibration, longer trajectories, and scene labels.

## 7. Required Conclusions

1. Current learned world model: `{conclusions['is_now_learned_world_model']}` for the synthetic 2.5D setting; real-data deployment remains a calibrated scaffold.
2. Synthetic t+100 reliability: {conclusions['synthetic_t100_reliability']}
3. Real-data t+12 reliability: {conclusions['real_t12_reliability']}
4. Real-data t+100: {conclusions['real_t100_status']}
5. Learned neural residual vs physics: {conclusions['learned_residual_vs_physics']}
6. SMC coverage: physics={conclusions['smc_coverage']['physics']}, neural={conclusions['smc_coverage']['neural']}, hybrid={conclusions['smc_coverage']['hybrid']}
7. Physical constraints: {conclusions['physical_constraints_effect']}
8. Terminal cluster diversity: {conclusions['terminal_cluster_diversity']}
9. Largest failure case: {conclusions['largest_failure_case']}
10. Next real data needed: {', '.join(conclusions['next_real_data_needed'])}

## 8. Do Not Overclaim

- Synthetic `t+100` is valid because the simulator supplies true future states.
- AerialMPT `t+100` remains trend-only free-run.
- If the neural proposal underperforms physics in any metric, that is a real result, not a reporting failure.
- Projection reduces violations, but it can also hide bad proposals by correcting them; projection cost and violation metrics must stay visible.
"""


def horizon_comparison_table(summary: Dict) -> str:
    rows = []
    for method, metrics in summary.items():
        for h in HORIZONS:
            row = {"proposal": method, "horizon": h}
            row.update(metrics["horizons"][str(h)])
            rows.append(row)
    return markdown_table(rows)


def proposal_scalar_table(summary: Dict) -> str:
    keys = [
        "collision_violation_rate",
        "obstacle_violation_rate",
        "boundary_violation_rate",
        "max_speed_violation_rate",
        "acceleration_violation_rate",
        "trajectory_smoothness_mean_jerk",
        "coverage_t100_fde_lt_2m",
        "endpoint_nll_t100",
        "cluster_diversity_score",
        "semantic_event_accuracy_top_cluster",
    ]
    rows = []
    for method, metrics in summary.items():
        row = {"proposal": method}
        for key in keys:
            row[key] = metrics.get(key)
        rows.append(row)
    return markdown_table(rows)


def terminal_cluster_tables(clusters: Dict) -> str:
    sections = []
    for method, payload in clusters.items():
        sections.append(f"### {method}\n\n{markdown_table(payload['clusters'])}")
    return "\n\n".join(sections)


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._"
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                value = round(value, 5)
            if isinstance(value, list):
                value = ", ".join(str(x) for x in value[:8])
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize_dataset(dataset: List[Dict]) -> Dict:
    splits = {}
    for split in ["train", "val", "test"]:
        episodes = [ep for ep in dataset if ep["meta"]["split"] == split]
        splits[f"{split}_episodes"] = len(episodes)
        splits[f"{split}_agents_mean"] = round(float(np.mean([ep["meta"]["agents"] for ep in episodes])), 2) if episodes else 0
    return {
        "name": "SyntheticPhysicalCrowd2.5D",
        "episodes": len(dataset),
        "frames_per_episode": EPISODE_FRAMES,
        "dt_seconds": DT,
        "agents_per_episode_min": int(min(ep["meta"]["agents"] for ep in dataset)),
        "agents_per_episode_max": int(max(ep["meta"]["agents"] for ep in dataset)),
        "state_columns": ", ".join(STATE_COLUMNS),
        **splits,
        "scene_templates": ", ".join(sorted(set(ep["meta"]["scene_name"] for ep in dataset))),
        "storage": str(DATASET_DIR),
    }


def key_console_summary(report: Dict) -> Dict:
    synthetic = report["synthetic_evaluation"]
    return {
        "synthetic_t100": {
            method: {
                "ADE_m": synthetic[method]["horizons"]["100"]["ADE_m"],
                "FDE_m": synthetic[method]["horizons"]["100"]["FDE_m"],
                "best_of_64_FDE_m": synthetic[method]["horizons"]["100"]["best_of_64_FDE_m"],
                "coverage": synthetic[method]["coverage_t100_fde_lt_2m"],
                "event_acc": synthetic[method]["semantic_event_accuracy_top_cluster"],
            }
            for method in synthetic
        },
        "aerialmpt": report["aerialmpt_real_data"].get("t100_status"),
        "report": str(REPORT_DIR / "report_stage2.md"),
    }


# Geometry and feature utilities


def normalize_array(x: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    return (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)


def denormalize_array(x: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    return x * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)


def clip_vectors(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(1e-6, norms))
    return vectors * scale


def unit_vector_np(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def point_in_rect(point: Tuple[float, float], rect: Rect, pad: float = 0.0) -> bool:
    return rect.x1 - pad <= point[0] <= rect.x2 + pad and rect.y1 - pad <= point[1] <= rect.y2 + pad


def point_in_any_rect(point: Tuple[float, float], rects: Sequence[Rect], pad: float = 0.0) -> bool:
    return any(point_in_rect(point, rect, pad) for rect in rects)


def push_out_rect(point: Tuple[float, float], rect: Rect, pad: float = 0.0) -> Tuple[float, float]:
    x, y = point
    distances = [
        (abs(x - (rect.x1 - pad)), (rect.x1 - pad, y)),
        (abs(x - (rect.x2 + pad)), (rect.x2 + pad, y)),
        (abs(y - (rect.y1 - pad)), (x, rect.y1 - pad)),
        (abs(y - (rect.y2 + pad)), (x, rect.y2 + pad)),
    ]
    return min(distances, key=lambda item: item[0])[1]


def nearest_point_rect(point: Tuple[float, float], rect: Rect) -> Tuple[float, float]:
    return float(np.clip(point[0], rect.x1, rect.x2)), float(np.clip(point[1], rect.y1, rect.y2))


def nearest_obstacle_vector(point: Tuple[float, float], scene: SceneSpec) -> Tuple[np.ndarray, float, bool]:
    best_vec = np.array([1.0, 0.0], dtype=np.float32)
    best_dist = 1e9
    inside_any = False
    for rect in scene.obstacles:
        nearest = nearest_point_rect(point, rect)
        vec = np.array([point[0] - nearest[0], point[1] - nearest[1]], dtype=np.float32)
        dist = float(np.linalg.norm(vec))
        inside = point_in_rect(point, rect)
        if inside:
            inside_any = True
            center = np.array([(rect.x1 + rect.x2) / 2, (rect.y1 + rect.y2) / 2], dtype=np.float32)
            vec = np.array(point, dtype=np.float32) - center
            dist = max(1e-5, float(np.linalg.norm(vec)))
        if dist < best_dist:
            best_dist = dist
            best_vec = vec / max(1e-5, dist)
    if best_dist == 1e9:
        return np.array([1.0, 0.0], dtype=np.float32), 99.0, False
    return best_vec.astype(np.float32), best_dist, inside_any


def obstacle_count_near(point: Tuple[float, float], scene: SceneSpec, radius: float) -> int:
    return sum(1 for rect in scene.obstacles if distance_tuple(point, nearest_point_rect(point, rect)) < radius)


def boundary_force(point: Tuple[float, float], scene: SceneSpec, margin: float) -> Tuple[float, float]:
    x, y = point
    force = np.zeros(2, dtype=np.float32)
    if x < margin:
        force[0] += margin - x
    if scene.width - x < margin:
        force[0] -= margin - (scene.width - x)
    if y < margin:
        force[1] += margin - y
    if scene.height - y < margin:
        force[1] -= margin - (scene.height - y)
    return float(force[0] * 0.9), float(force[1] * 0.9)


def distance_tuple(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def min_gap_and_collisions(state: np.ndarray) -> Tuple[float, int]:
    min_gap = 999.0
    collisions = 0
    for i in range(state.shape[0]):
        for j in range(i + 1, state.shape[0]):
            gap = float(np.linalg.norm(state[i, :2] - state[j, :2]) - (state[i, 6] + state[j, 6]))
            min_gap = min(min_gap, gap)
            if gap < -1e-4:
                collisions += 1
    return min_gap, collisions


def local_density_for_agent(state: np.ndarray, index: int) -> float:
    distances = np.linalg.norm(state[:, :2] - state[index, :2], axis=1)
    return float(np.sum((distances < 2.0) & (distances > 0.0)))


def local_density_values(state: np.ndarray) -> np.ndarray:
    return np.array([local_density_for_agent(state, i) for i in range(state.shape[0])], dtype=np.float32)


def density_people_per_m2_np(state: np.ndarray) -> float:
    if state.shape[0] <= 1:
        return 0.0
    min_xy = state[:, :2].min(axis=0)
    max_xy = state[:, :2].max(axis=0)
    area = max(1.0, float(np.prod(max_xy - min_xy)))
    return float(state.shape[0] / area)


def time_to_collision(p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray, radius: float) -> float:
    dp = p2 - p1
    dv = v2 - v1
    a = float(np.dot(dv, dv))
    b = 2.0 * float(np.dot(dp, dv))
    c = float(np.dot(dp, dp) - radius * radius)
    if a < 1e-8:
        return 8.0
    disc = b * b - 4 * a * c
    if disc < 0:
        return 8.0
    t = (-b - math.sqrt(disc)) / (2 * a)
    return float(t if t > 0 else 8.0)


def speed_accel_penalty(state: np.ndarray) -> float:
    speed = np.linalg.norm(state[:, 2:4], axis=1)
    accel = np.linalg.norm(state[:, 4:6], axis=1)
    return float(np.sum(np.maximum(0.0, speed - MAX_SPEED)) + 0.4 * np.sum(np.maximum(0.0, accel - MAX_ACCEL)))


def normalize_particle_list(particles: List[Dict]) -> List[Dict]:
    max_log = max(p["log_weight"] for p in particles)
    weights = np.array([math.exp(p["log_weight"] - max_log) for p in particles], dtype=np.float64)
    weights /= max(1e-12, float(weights.sum()))
    for particle, weight in zip(particles, weights):
        particle["weight"] = float(weight)
    return particles


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    return weights / max(1e-12, float(weights.sum()))


def effective_sample_size(weights: np.ndarray) -> float:
    w = weights / max(1e-12, float(weights.sum()))
    return float(1.0 / max(1e-12, float(np.sum(w * w))))


def systematic_resample_indexes(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = weights / max(1e-12, float(weights.sum()))
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    return np.searchsorted(cumulative, positions, side="left")


def copy_particle(particle: Dict) -> Dict:
    return {"history": particle["history"].copy(), "log_weight": 0.0, "projection_cost": particle.get("projection_cost", 0.0), "weight": 1.0 / PARTICLES}


def physical_score(trajectory: np.ndarray, scene: SceneSpec) -> float:
    bad = 0
    total = trajectory.shape[0]
    for frame in trajectory:
        min_gap, collisions = min_gap_and_collisions(frame)
        obstacle = any(point_in_any_rect(tuple(agent[:2]), scene.obstacles, pad=float(agent[6])) for agent in frame)
        boundary = any(agent[0] < agent[6] or agent[0] > scene.width - agent[6] or agent[1] < agent[6] or agent[1] > scene.height - agent[6] for agent in frame)
        if collisions or min_gap < -1e-4 or obstacle or boundary:
            bad += 1
    return bad / max(1, total)


def first_reach_time(trajectory: np.ndarray) -> float:
    reach_times = []
    for i in range(trajectory.shape[1]):
        distances = np.linalg.norm(trajectory[:, i, :2] - trajectory[:, i, 7:9], axis=1)
        hits = np.where(distances < 1.4)[0]
        if len(hits):
            reach_times.append(float(hits[0]))
    return float(np.mean(reach_times)) if reach_times else float(trajectory.shape[0])


def naming_reason(label: str) -> str:
    reasons = {
        "smooth-pass": "Most agents reach an exit with moderate speed and low violation rates.",
        "congestion": "Low-speed intervals and high-density intervals dominate the trajectory.",
        "detour": "Path length is substantially longer than straight-line displacement.",
        "stalled": "Agents spend a large fraction of time nearly stopped.",
        "split-flow": "Final spatial spread indicates the crowd separated into multiple flows.",
        "collision-risk": "Minimum gap is close to or below the combined body radii.",
        "partial-flow": "Some agents move toward goals but exit completion is incomplete.",
    }
    return reasons.get(label, "Semantic label assigned from terminal event features.")


def scene_to_dict(scene: SceneSpec) -> Dict:
    return {
        "name": scene.name,
        "width": scene.width,
        "height": scene.height,
        "obstacles": [rect.__dict__ for rect in scene.obstacles],
        "exits": scene.exits,
        "spawn_regions": scene.spawn_regions,
        "event_hint": scene.event_hint,
    }


def scene_from_dict(data: Dict) -> SceneSpec:
    return SceneSpec(
        name=data["name"],
        width=float(data["width"]),
        height=float(data["height"]),
        obstacles=[Rect(**rect) for rect in data["obstacles"]],
        exits={key: tuple(value) for key, value in data["exits"].items()},
        spawn_regions=data["spawn_regions"],
        event_hint=data["event_hint"],
    )


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


if __name__ == "__main__":
    main()
