#!/usr/bin/env python3
"""Top-down human trajectory world model with collision-aware branching.

Data: AerialMPT real aerial pedestrian images + point tracks.
Goal: train a lightweight character trajectory model, represent each person as
a visual point on the real image plus a physical collision body, and run t+100
multi-branch prediction with reward/penalty scoring.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from PIL import Image, ImageDraw, ImageFont
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "aerialmpt" / "extracted"
OUT_DIR = ROOT / "experiments" / "outputs" / "human_collision_world_model"
FPS = 2.0
DT = 1.0 / FPS
HORIZON = 100
PERSON_MASS_KG = 70.0
DEFAULT_RADIUS_PX = 6.0


@dataclass
class SocialNonlinearModel:
    regressor: MLPRegressor
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    residual_std: np.ndarray
    metrics: Dict[str, float]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_tracks = load_aerialmpt(DATA_ROOT)
    train_tracks = all_tracks[all_tracks["split"] == "train"].copy()
    test_tracks = all_tracks[all_tracks["split"] == "test"].copy()
    train_rows = build_transition_rows(train_tracks)
    test_rows = build_transition_rows(test_tracks)
    model = train_model(train_rows, test_rows)

    scene_name = choose_holdout_scene(test_tracks)
    scene_tracks = test_tracks[test_tracks["scene"] == scene_name].copy()
    start_frame = choose_start_frame(scene_tracks)
    initial_state = select_active_agents(state_at_frame(scene_tracks, start_frame), max_agents=18)
    tracked_ids = {agent["id"] for agent in initial_state}
    actual_state = [agent for agent in state_at_frame(scene_tracks, start_frame + HORIZON) if agent["id"] in tracked_ids]
    prediction = smc_rollout(initial_state, model, scene_tracks, HORIZON, seed=42)
    actual_summary = summarize_state(actual_state)
    prediction["actual_t100"] = actual_summary
    prediction["dataset"] = {
        "source": "AerialMPT official DLR dataset",
        "scenes": int(all_tracks["scene"].nunique()),
        "rows": int(len(all_tracks)),
        "characters": int(all_tracks[["scene", "track_id"]].drop_duplicates().shape[0]),
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "holdout_scene": scene_name,
        "start_frame": int(start_frame),
        "fps": FPS,
        "horizon_steps": HORIZON,
        "modeled_agents": len(initial_state),
    }
    prediction["model"] = model.metrics
    prediction["physics"] = {
        "body": "top-down capsule/circle approximation",
        "mass_kg": PERSON_MASS_KG,
        "collision_radius_px": DEFAULT_RADIUS_PX,
        "collision_constraint": "explicit projection pushes overlapping bodies apart after every sampled transition",
        "collision_weight": "importance weight penalizes projected displacement and near-contact states",
        "boundary_penalty": "negative reward when predicted body leaves image bounds",
    }

    summary_path = OUT_DIR / "summary.json"
    image_path = OUT_DIR / "rollout.png"
    model_path = OUT_DIR / "nonlinear_social_model.joblib"
    summary_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    dump({"model": model, "features": feature_names(), "targets": ["dx", "dy"]}, model_path)
    render_rollout(scene_tracks, start_frame, initial_state, prediction, actual_state, image_path)

    print(json.dumps({"summary": str(summary_path), "image": str(image_path), "model": str(model_path)}, indent=2))
    print(json.dumps(key_summary(prediction), indent=2))


def load_aerialmpt(root: Path) -> pd.DataFrame:
    rows = []
    for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        split = split_dir.name
        for scene_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            gts_files = list(scene_dir.glob("*_gts.txt"))
            if not gts_files:
                continue
            image_names = scene_image_names(scene_dir)
            image_size = first_image_size(scene_dir, image_names)
            scene = scene_dir.name
            for line in gts_files[0].read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                rows.append(
                    {
                        "split": split,
                        "scene": scene,
                        "frame": frame,
                        "track_id": track_id,
                        "x": x,
                        "y": y,
                        "w": width,
                        "h": height,
                        "radius": max(DEFAULT_RADIUS_PX, max(width, height) * 1.5),
                        "image_width": image_size[0],
                        "image_height": image_size[1],
                        "image_name": image_names[frame - 1] if frame - 1 < len(image_names) else None,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No AerialMPT tracks found under {root}")
    return df.sort_values(["split", "scene", "track_id", "frame"]).reset_index(drop=True)


def scene_image_names(scene_dir: Path) -> List[str]:
    list_file = scene_dir / "image_list.txt"
    if list_file.exists():
        return [line.strip() for line in list_file.read_text().splitlines() if line.strip()]
    return [path.stem for path in sorted(scene_dir.glob("*.png"))]


def first_image_size(scene_dir: Path, image_names: List[str]) -> Tuple[int, int]:
    candidates = [scene_dir / f"{name}.png" for name in image_names]
    candidates += sorted(scene_dir.glob("*.png"))
    for path in candidates:
        if path.exists():
            with Image.open(path) as image:
                return image.size
    return (512, 512)


def build_transition_rows(df: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for _, group in df.groupby(["scene", "track_id"], sort=False):
        group = group.sort_values("frame").copy()
        group["prev_x"] = group["x"].shift(1)
        group["prev_y"] = group["y"].shift(1)
        group["prev_frame"] = group["frame"].shift(1)
        group["next_x"] = group["x"].shift(-1)
        group["next_y"] = group["y"].shift(-1)
        group["next_frame"] = group["frame"].shift(-1)
        group["dt_prev"] = ((group["frame"] - group["prev_frame"]) / FPS).fillna(DT)
        group["dt_next"] = (group["next_frame"] - group["frame"]) / FPS
        group = group[(group["dt_next"] > 0) & (group["dt_next"] <= 1.5)].copy()
        group["vx"] = ((group["x"] - group["prev_x"]) / group["dt_prev"]).fillna(0)
        group["vy"] = ((group["y"] - group["prev_y"]) / group["dt_prev"]).fillna(0)
        group["dx"] = group["next_x"] - group["x"]
        group["dy"] = group["next_y"] - group["y"]
        groups.append(group)

    rows = pd.concat(groups, ignore_index=True)
    rows = add_social_features(rows)
    return rows.replace([np.inf, -np.inf], np.nan).dropna(subset=["dx", "dy"])


def add_social_features(rows: pd.DataFrame) -> pd.DataFrame:
    nearest_dx = np.zeros(len(rows))
    nearest_dy = np.zeros(len(rows))
    nearest_dist = np.full(len(rows), 999.0)
    local_density = np.zeros(len(rows))
    social_flow_vx = np.zeros(len(rows))
    social_flow_vy = np.zeros(len(rows))
    separation_x = np.zeros(len(rows))
    separation_y = np.zeros(len(rows))
    closing_speed = np.zeros(len(rows))

    for _, frame_rows in rows.groupby(["scene", "frame"], sort=False):
        idxs = frame_rows.index.to_numpy()
        points = frame_rows[["x", "y"]].to_numpy(float)
        velocities = frame_rows[["vx", "vy"]].to_numpy(float) if {"vx", "vy"}.issubset(frame_rows.columns) else np.zeros_like(points)
        if len(points) <= 1:
            continue
        diff = points[:, None, :] - points[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        nearest = dist.argmin(axis=1)
        nearest_dist[idxs] = dist[np.arange(len(points)), nearest]
        nearest_vec = points[nearest] - points
        nearest_dx[idxs] = nearest_vec[:, 0]
        nearest_dy[idxs] = nearest_vec[:, 1]
        neighbor_mask = dist < 36
        local_density[idxs] = neighbor_mask.sum(axis=1)
        for row_index in range(len(points)):
            neighbors = np.where(neighbor_mask[row_index])[0]
            if len(neighbors) == 0:
                continue
            social_flow_vx[idxs[row_index]] = velocities[neighbors, 0].mean() - velocities[row_index, 0]
            social_flow_vy[idxs[row_index]] = velocities[neighbors, 1].mean() - velocities[row_index, 1]
            push = points[row_index] - points[neighbors]
            inv_dist = 1 / np.maximum(1.0, dist[row_index, neighbors])
            separation = (push * inv_dist.reshape(-1, 1)).mean(axis=0)
            separation_x[idxs[row_index]] = separation[0]
            separation_y[idxs[row_index]] = separation[1]
            rel_v = velocities[row_index] - velocities[nearest[row_index]]
            unit_to_nearest = nearest_vec[row_index] / max(1.0, nearest_dist[idxs[row_index]])
            closing_speed[idxs[row_index]] = -float(np.dot(rel_v, unit_to_nearest))

    rows = rows.copy()
    rows["nearest_dx"] = nearest_dx
    rows["nearest_dy"] = nearest_dy
    rows["nearest_dist"] = nearest_dist
    rows["local_density"] = local_density
    rows["social_flow_vx"] = social_flow_vx
    rows["social_flow_vy"] = social_flow_vy
    rows["separation_x"] = separation_x
    rows["separation_y"] = separation_y
    rows["closing_speed"] = closing_speed
    rows["left_wall"] = rows["x"]
    rows["right_wall"] = rows["image_width"] - rows["x"]
    rows["top_wall"] = rows["y"]
    rows["bottom_wall"] = rows["image_height"] - rows["y"]
    return rows


def train_model(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> SocialNonlinearModel:
    x_train = make_features(train_rows)
    y_train = make_targets(train_rows)
    x_test = make_features(test_rows)
    y_test = make_targets(test_rows)
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train_n = feature_scaler.fit_transform(x_train)
    y_train_n = target_scaler.fit_transform(y_train)
    x_test_n = feature_scaler.transform(x_test)

    regressor = MLPRegressor(
        hidden_layer_sizes=(96, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0008,
        batch_size=512,
        learning_rate_init=0.0025,
        max_iter=380,
        early_stopping=True,
        validation_fraction=0.14,
        n_iter_no_change=18,
        random_state=11,
    )
    regressor.fit(x_train_n, y_train_n)
    pred_train = target_scaler.inverse_transform(regressor.predict(x_train_n))
    pred_test = target_scaler.inverse_transform(regressor.predict(x_test_n))
    residual = y_train - pred_train
    residual_std = np.clip(residual.std(axis=0), 0.6, 16.0)
    rmse = np.sqrt(((y_test - pred_test) ** 2).mean(axis=0))
    ade = float(np.linalg.norm(y_test - pred_test, axis=1).mean())
    collision_rate = estimate_collision_rate(test_rows)

    return SocialNonlinearModel(
        regressor=regressor,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        residual_std=residual_std,
        metrics={
            "model_type": "social-message MLP transition model",
            "train_transitions": int(len(train_rows)),
            "test_transitions": int(len(test_rows)),
            "dx_rmse_px": float(rmse[0]),
            "dy_rmse_px": float(rmse[1]),
            "ade_px": ade,
            "residual_dx_px": float(residual_std[0]),
            "residual_dy_px": float(residual_std[1]),
            "observed_collision_rate": collision_rate,
            "mlp_iterations": int(regressor.n_iter_),
            "mlp_loss": float(regressor.loss_),
        },
    )


def make_features(rows: pd.DataFrame) -> np.ndarray:
    width = rows["image_width"].to_numpy(float)
    height = rows["image_height"].to_numpy(float)
    nearest_dist = np.minimum(rows["nearest_dist"].to_numpy(float), 200)
    return np.column_stack(
        [
            np.ones(len(rows)),
            rows["x"].to_numpy(float) / width,
            rows["y"].to_numpy(float) / height,
            rows["vx"].to_numpy(float) / 80,
            rows["vy"].to_numpy(float) / 80,
            rows["nearest_dx"].to_numpy(float) / width,
            rows["nearest_dy"].to_numpy(float) / height,
            nearest_dist / np.maximum(width, height),
            rows["local_density"].to_numpy(float) / 12,
            rows["social_flow_vx"].to_numpy(float) / 80,
            rows["social_flow_vy"].to_numpy(float) / 80,
            rows["separation_x"].to_numpy(float),
            rows["separation_y"].to_numpy(float),
            rows["closing_speed"].to_numpy(float) / 80,
            rows["left_wall"].to_numpy(float) / width,
            rows["right_wall"].to_numpy(float) / width,
            rows["top_wall"].to_numpy(float) / height,
            rows["bottom_wall"].to_numpy(float) / height,
        ]
    )


def make_targets(rows: pd.DataFrame) -> np.ndarray:
    return rows[["dx", "dy"]].to_numpy(float)


def feature_names() -> List[str]:
    return [
        "bias",
        "x_norm",
        "y_norm",
        "vx_norm",
        "vy_norm",
        "nearest_dx_norm",
        "nearest_dy_norm",
        "nearest_dist_norm",
        "local_density",
        "social_flow_vx_norm",
        "social_flow_vy_norm",
        "separation_x",
        "separation_y",
        "closing_speed_norm",
        "left_wall_norm",
        "right_wall_norm",
        "top_wall_norm",
        "bottom_wall_norm",
    ]


def choose_holdout_scene(test_tracks: pd.DataFrame) -> str:
    counts = test_tracks.groupby("scene")["track_id"].nunique().sort_values(ascending=False)
    return str(counts.index[0])


def choose_start_frame(scene_tracks: pd.DataFrame) -> int:
    frames = sorted(scene_tracks["frame"].unique())
    valid = [frame for frame in frames if frame + HORIZON <= max(frames)]
    return valid[len(valid) // 2] if valid else frames[0]


def state_at_frame(scene_tracks: pd.DataFrame, frame: int) -> List[Dict]:
    current = scene_tracks[scene_tracks["frame"] == frame].copy()
    if current.empty:
        idxs = (scene_tracks["frame"] - frame).abs().groupby(scene_tracks["track_id"]).idxmin()
        current = scene_tracks.loc[idxs].copy()
    state = []
    for _, row in current.iterrows():
        history = scene_tracks[(scene_tracks["track_id"] == row["track_id"]) & (scene_tracks["frame"] <= row["frame"])].sort_values("frame").tail(2)
        vx, vy = 0.0, 0.0
        if len(history) == 2:
            before, now = history.iloc[0], history.iloc[1]
            dt = max(DT, (now["frame"] - before["frame"]) / FPS)
            vx = float((now["x"] - before["x"]) / dt)
            vy = float((now["y"] - before["y"]) / dt)
        state.append(
            {
                "id": int(row["track_id"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": 0.0,
                "vx": vx,
                "vy": vy,
                "radius": float(row["radius"]),
                "mass": PERSON_MASS_KG,
                "image_width": int(row["image_width"]),
                "image_height": int(row["image_height"]),
            }
        )
    return state


def select_active_agents(state: List[Dict], max_agents: int) -> List[Dict]:
    if len(state) <= max_agents:
        return state
    center = np.array([[agent["x"], agent["y"]] for agent in state], dtype=float).mean(axis=0)

    def score(agent: Dict) -> float:
        speed = math.hypot(agent["vx"], agent["vy"])
        distance_to_center = math.hypot(agent["x"] - center[0], agent["y"] - center[1])
        return speed * 2.5 - distance_to_center * 0.02

    return sorted(state, key=score, reverse=True)[:max_agents]


def smc_rollout(initial_state: List[Dict], model: SocialNonlinearModel, scene_tracks: pd.DataFrame, horizon: int, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    particle_count = 160
    proposal_count = 3
    particles = [
        {
            "state": clone_state(initial_state),
            "logp": 0.0,
            "reward": 0.0,
            "projection_cost": 0.0,
            "trace": compact_state(initial_state),
        }
        for _ in range(particle_count)
    ]
    ess_history = []

    for step in range(horizon):
        proposals = []
        for particle in particles:
            base = predict_deltas(particle["state"], model)
            for _ in range(proposal_count):
                sampled_delta, proposal_logp = sample_proposal(base, model, rng)
                next_state, projection_cost = apply_deltas(particle["state"], sampled_delta)
                constraint_score = physics_score(next_state, projection_cost)
                reward = particle["reward"] + constraint_score
                proposals.append(
                    {
                        "state": next_state,
                        "logp": particle["logp"] + proposal_logp + constraint_score * 0.55,
                        "reward": reward,
                        "projection_cost": particle["projection_cost"] + projection_cost,
                        "trace": append_trace(particle["trace"], next_state, step, horizon),
                    }
                )

        normalized = normalize_paths(proposals)
        ess = 1.0 / sum(max(1e-12, particle["probability"]) ** 2 for particle in normalized)
        ess_history.append(round(float(ess), 3))
        particles = systematic_resample(normalized, particle_count, rng) if ess < particle_count * 0.62 else normalized[:particle_count]

    weighted = normalize_paths(particles)
    return {
        "config": {
            "rollout": "SMC importance sampling with systematic resampling",
            "horizon": horizon,
            "particles": particle_count,
            "proposal_count": proposal_count,
            "paths_kept": len(weighted),
            "mean_ess": round(float(np.mean(ess_history)), 3),
            "min_ess": round(float(np.min(ess_history)), 3),
        },
        "outcomes": cluster_outcomes(weighted),
        "top_paths": [
            {
                "probability": round(path["probability"], 5),
                "reward": round(path["reward"], 3),
                "projection_cost": round(path["projection_cost"], 3),
                "terminal": summarize_state(path["state"]),
                "trace": path["trace"],
            }
            for path in weighted[:8]
        ],
    }


def predict_deltas(state: List[Dict], model: SocialNonlinearModel) -> np.ndarray:
    rows = state_to_feature_rows(state)
    features = make_features(rows)
    normalized = model.feature_scaler.transform(features)
    return model.target_scaler.inverse_transform(model.regressor.predict(normalized))


def state_to_feature_rows(state: List[Dict]) -> pd.DataFrame:
    rows = pd.DataFrame(state)
    rows["frame"] = 0
    rows["track_id"] = rows["id"]
    rows["w"] = rows["radius"] / 1.5
    rows["h"] = rows["radius"] / 1.5
    rows["split"] = "rollout"
    rows["scene"] = "rollout"
    return add_social_features(rows)


def sample_proposal(base: np.ndarray, model: SocialNonlinearModel, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    std = np.maximum(model.residual_std, 0.35)
    noise = rng.normal(0, std.reshape(1, 2), size=base.shape)
    z = noise / std.reshape(1, 2)
    proposal_logp = -0.5 * float(np.mean(z**2))
    return base + noise, proposal_logp


def apply_deltas(state: List[Dict], deltas: np.ndarray) -> Tuple[List[Dict], float]:
    next_state = []
    for agent, delta in zip(state, deltas):
        width, height = agent["image_width"], agent["image_height"]
        x = float(np.clip(agent["x"] + delta[0], 0, width))
        y = float(np.clip(agent["y"] + delta[1], 0, height))
        next_agent = dict(agent)
        next_agent["vx"] = (x - agent["x"]) / DT
        next_agent["vy"] = (y - agent["y"]) / DT
        next_agent["x"] = x
        next_agent["y"] = y
        next_agent["z"] = float(min(24.0, math.hypot(next_agent["vx"], next_agent["vy"]) * 0.05))
        next_state.append(next_agent)
    return project_collision_constraints(state, next_state)


def project_collision_constraints(previous_state: List[Dict], state: List[Dict], iterations: int = 5) -> Tuple[List[Dict], float]:
    projection_cost = 0.0
    projected = [dict(agent) for agent in state]

    for _ in range(iterations):
        moved = False
        for i, a in enumerate(projected):
            for b in projected[i + 1 :]:
                dx = b["x"] - a["x"]
                dy = b["y"] - a["y"]
                dist = math.hypot(dx, dy)
                min_dist = a["radius"] + b["radius"]
                if dist >= min_dist:
                    continue
                moved = True
                if dist < 1e-6:
                    dx, dy, dist = 1.0, 0.0, 1.0
                overlap = min_dist - dist
                nx, ny = dx / dist, dy / dist
                inv_mass_a = 1 / max(1.0, a["mass"])
                inv_mass_b = 1 / max(1.0, b["mass"])
                total_inv_mass = inv_mass_a + inv_mass_b
                push_a = overlap * (inv_mass_a / total_inv_mass)
                push_b = overlap * (inv_mass_b / total_inv_mass)
                a["x"] -= nx * push_a
                a["y"] -= ny * push_a
                b["x"] += nx * push_b
                b["y"] += ny * push_b
                projection_cost += overlap
        if not moved:
            break

    for before, agent in zip(previous_state, projected):
        radius = agent["radius"]
        clamped_x = float(np.clip(agent["x"], radius, agent["image_width"] - radius))
        clamped_y = float(np.clip(agent["y"], radius, agent["image_height"] - radius))
        projection_cost += abs(clamped_x - agent["x"]) + abs(clamped_y - agent["y"])
        agent["x"] = clamped_x
        agent["y"] = clamped_y
        agent["vx"] = (agent["x"] - before["x"]) / DT
        agent["vy"] = (agent["y"] - before["y"]) / DT
        agent["z"] = float(min(24.0, math.hypot(agent["vx"], agent["vy"]) * 0.05))

    return projected, projection_cost


def physics_score(state: List[Dict], projection_cost: float) -> float:
    score = -0.075 * projection_cost
    for i, a in enumerate(state):
        if a["x"] <= a["radius"] or a["x"] >= a["image_width"] - a["radius"]:
            score -= 0.5
        if a["y"] <= a["radius"] or a["y"] >= a["image_height"] - a["radius"]:
            score -= 0.5
        speed = math.hypot(a["vx"], a["vy"])
        if speed > 90:
            score -= (speed - 90) / 80
        for b in state[i + 1 :]:
            dist = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            min_dist = a["radius"] + b["radius"]
            if dist < min_dist:
                score -= 2.2 * (1 - dist / max(1, min_dist))
            elif dist < min_dist * 1.8:
                score -= 0.16 * (1 - dist / (min_dist * 1.8))
    if minimum_gap(state) > 2:
        score += 0.08
    return score


def estimate_collision_rate(rows: pd.DataFrame) -> float:
    collisions = 0
    total_pairs = 0
    for _, frame_rows in rows.groupby(["scene", "frame"], sort=False):
        points = frame_rows[["x", "y"]].to_numpy(float)
        radii = frame_rows["radius"].to_numpy(float)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                total_pairs += 1
                if np.linalg.norm(points[i] - points[j]) < radii[i] + radii[j]:
                    collisions += 1
    return float(collisions / total_pairs) if total_pairs else 0.0


def summarize_state(state: List[Dict]) -> Dict:
    if not state:
        return {"agents": 0}
    points = np.array([[a["x"], a["y"]] for a in state])
    speeds = np.array([math.hypot(a["vx"], a["vy"]) for a in state])
    min_gap = minimum_gap(state)
    collisions = sum(1 for gap in pair_gaps(state) if gap < 0)
    center = points.mean(axis=0)
    spread = np.linalg.norm(points - center, axis=1).mean() if len(points) > 1 else 0.0
    if collisions:
        label = "collision-risk cluster"
    elif spread > 85:
        label = "crowd disperses"
    elif speeds.mean() < 5:
        label = "crowd slows or stalls"
    else:
        label = "crowd continues coordinated flow"
    return {
        "agents": len(state),
        "center": [round(float(center[0]), 2), round(float(center[1]), 2)],
        "mean_speed_px_s": round(float(speeds.mean()), 3),
        "spread_px": round(float(spread), 3),
        "min_gap_px": round(float(min_gap), 3),
        "collisions": collisions,
        "label": label,
    }


def cluster_outcomes(paths: List[Dict]) -> List[Dict]:
    clusters: Dict[str, Dict] = {}
    for path in paths:
        terminal = summarize_state(path["state"])
        key = "|".join(
            [
                terminal["label"],
                str(int(terminal["center"][0] // 45)),
                str(int(terminal["center"][1] // 45)),
                str(min(5, terminal["collisions"])),
            ]
        )
        cluster = clusters.setdefault(key, {"probability": 0.0, "paths": 0, "reward": 0.0, "representative": terminal, "best": 0.0})
        cluster["probability"] += path["probability"]
        cluster["paths"] += 1
        cluster["reward"] += path["probability"] * path["reward"]
        if path["probability"] > cluster["best"]:
            cluster["best"] = path["probability"]
            cluster["representative"] = terminal
    return [
        {
            "label": cluster["representative"]["label"],
            "probability": round(cluster["probability"], 4),
            "paths": cluster["paths"],
            "expected_reward": round(cluster["reward"], 3),
            "representative": cluster["representative"],
        }
        for cluster in sorted(clusters.values(), key=lambda value: value["probability"], reverse=True)[:6]
    ]


def render_rollout(scene_tracks: pd.DataFrame, start_frame: int, initial_state: List[Dict], prediction: Dict, actual_state: List[Dict], out_path: Path) -> None:
    scene_dir = DATA_ROOT / "test" / prediction["dataset"]["holdout_scene"] if "dataset" in prediction else None
    image_path = None
    if scene_dir and scene_dir.exists():
        image_name = scene_tracks[scene_tracks["frame"] == start_frame]["image_name"].dropna()
        if not image_name.empty:
            candidate = scene_dir / f"{image_name.iloc[0]}.png"
            if candidate.exists():
                image_path = candidate
    if image_path:
        image = Image.open(image_path).convert("RGB")
    else:
        width = int(scene_tracks["image_width"].iloc[0])
        height = int(scene_tracks["image_height"].iloc[0])
        image = Image.new("RGB", (width, height), "#eef2f5")
    scale = max(1, int(900 / max(image.size)))
    canvas = image.resize((image.width * scale, image.height * scale))
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()

    def p(point: Tuple[float, float]) -> Tuple[float, float]:
        return (point[0] * scale, point[1] * scale)

    for agent in initial_state:
        draw.ellipse(circle_box(*p((agent["x"], agent["y"])), agent["radius"] * scale), outline=(255, 255, 255, 220), width=2)
        draw.text((agent["x"] * scale + 6, agent["y"] * scale - 6), str(agent["id"]), fill=(255, 255, 255, 230), font=font)

    colors = [(15, 118, 110, 180), (226, 149, 53, 170), (47, 128, 237, 160), (143, 99, 168, 150)]
    for idx, path in enumerate(prediction["top_paths"][:6]):
        color = colors[idx % len(colors)]
        for agent_id, points in path["trace"].items():
            scaled = [p(tuple(point)) for point in points]
            if len(scaled) > 1:
                draw.line(scaled, fill=color, width=max(2, scale))
            if scaled:
                draw.ellipse(circle_box(*scaled[-1], 4 * scale), fill=color)

    for agent in actual_state:
        x, y = p((agent["x"], agent["y"]))
        draw.line((x - 6 * scale, y - 6 * scale, x + 6 * scale, y + 6 * scale), fill=(220, 38, 38, 230), width=max(2, scale))
        draw.line((x - 6 * scale, y + 6 * scale, x + 6 * scale, y - 6 * scale), fill=(220, 38, 38, 230), width=max(2, scale))

    panel_h = 122
    out = Image.new("RGB", (canvas.width, canvas.height + panel_h), "#f8fafc")
    out.paste(canvas, (0, 0))
    panel = ImageDraw.Draw(out, "RGBA")
    panel.rectangle((0, canvas.height, canvas.width, canvas.height + panel_h), fill=(248, 250, 252, 255))
    panel.text((10, canvas.height + 10), "Human Collision World Model: t+100 branch rollout", fill=(24, 33, 47), font=font)
    y = canvas.height + 30
    for outcome in prediction["outcomes"][:4]:
        text = f"p={outcome['probability']:.3f} reward={outcome['expected_reward']:.2f} {outcome['label']} min_gap={outcome['representative']['min_gap_px']}"
        panel.text((10, y), text, fill=(82, 97, 113), font=font)
        y += 18
    panel.text((10, canvas.height + 106), "white circles=start physical bodies, colored lines=predicted branches, red X=actual t+100 people", fill=(82, 97, 113), font=font)
    out.save(out_path)


def normalize_paths(paths: List[Dict]) -> List[Dict]:
    max_log = max(path["logp"] for path in paths)
    weights = [math.exp(path["logp"] - max_log) for path in paths]
    total = sum(weights) or 1.0
    for path, weight in zip(paths, weights):
        path["probability"] = weight / total
    return sorted(paths, key=lambda item: item["probability"], reverse=True)


def systematic_resample(paths: List[Dict], count: int, rng: np.random.Generator) -> List[Dict]:
    weights = np.array([path["probability"] for path in paths], dtype=float)
    weights = weights / max(1e-12, weights.sum())
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    indexes = np.searchsorted(cumulative, positions, side="left")
    resampled = []
    for index in indexes:
        path = paths[int(min(index, len(paths) - 1))]
        resampled.append(
            {
                "state": clone_state(path["state"]),
                "logp": 0.0,
                "probability": 1.0 / count,
                "reward": path["reward"],
                "projection_cost": path["projection_cost"],
                "trace": {agent_id: [list(point) for point in points] for agent_id, points in path["trace"].items()},
            }
        )
    return resampled


def compact_state(state: List[Dict]) -> Dict[str, List[List[float]]]:
    return {str(agent["id"]): [[round(agent["x"], 2), round(agent["y"], 2)]] for agent in state}


def append_trace(trace: Dict[str, List[List[float]]], state: List[Dict], step: int, horizon: int) -> Dict[str, List[List[float]]]:
    latest = {agent_id: list(points) for agent_id, points in trace.items()}
    if step % 8 == 0 or step == horizon - 1:
        for agent in state:
            latest.setdefault(str(agent["id"]), []).append([round(agent["x"], 2), round(agent["y"], 2)])
    return latest


def minimum_gap(state: List[Dict]) -> float:
    gaps = pair_gaps(state)
    return min(gaps) if gaps else 999.0


def pair_gaps(state: List[Dict]) -> List[float]:
    gaps = []
    for i, a in enumerate(state):
        for b in state[i + 1 :]:
            gaps.append(math.hypot(a["x"] - b["x"], a["y"] - b["y"]) - (a["radius"] + b["radius"]))
    return gaps


def circle_box(x: float, y: float, r: float) -> Tuple[float, float, float, float]:
    return (x - r, y - r, x + r, y + r)


def clone_state(state: List[Dict]) -> List[Dict]:
    return [dict(agent) for agent in state]


def key_summary(prediction: Dict) -> Dict:
    return {
        "dataset": prediction["dataset"],
        "model": prediction["model"],
        "physics": prediction["physics"],
        "outcomes": prediction["outcomes"],
        "actual_t100": prediction["actual_t100"],
    }


if __name__ == "__main__":
    main()
