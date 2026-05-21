#!/usr/bin/env python3
"""Sequence Transformer world model with SMC and hard collision constraints.

This is the heavier nonlinear experiment for top-down human trajectory
prediction. It trains a small temporal Transformer over AerialMPT tracks, then
uses sequential Monte Carlo to branch to t+100 while projecting every sampled
state back into an explicit non-overlap physical constraint.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from human_collision_world_model import (
    DATA_ROOT,
    FPS,
    HORIZON,
    PERSON_MASS_KG,
    add_social_features,
    append_trace,
    apply_deltas,
    choose_holdout_scene,
    choose_start_frame,
    clone_state,
    cluster_outcomes,
    compact_state,
    estimate_collision_rate,
    feature_names,
    key_summary,
    load_aerialmpt,
    make_features,
    make_targets,
    normalize_paths,
    physics_score,
    render_rollout,
    select_active_agents,
    state_at_frame,
    state_to_feature_rows,
    summarize_state,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "outputs" / "human_transformer_smc_world_model"
OBS_STEPS = 8
BATCH_SIZE = 768
MAX_EPOCHS = 14
PATIENCE = 4


@dataclass
class TorchSequenceWorldModel:
    network: "SocialTransformer"
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    residual_std: np.ndarray
    metrics: Dict[str, float]
    obs_steps: int
    feature_dim: int


class SocialTransformer(nn.Module):
    def __init__(self, feature_dim: int, obs_steps: int, d_model: int = 48) -> None:
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, obs_steps, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=96,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 56),
            nn.GELU(),
            nn.Linear(56, 2),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(sequence) + self.position[:, : sequence.shape[1], :]
        encoded = self.encoder(hidden)
        return self.head(encoded[:, -1, :])


def main() -> None:
    seed_everything(17)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tracks = load_aerialmpt(DATA_ROOT)
    train_tracks = all_tracks[all_tracks["split"] == "train"].copy()
    test_tracks = all_tracks[all_tracks["split"] == "test"].copy()
    train_rows = build_transition_rows(train_tracks)
    test_rows = build_transition_rows(test_tracks)
    model = train_transformer_model(train_rows, test_rows)

    scene_name = choose_holdout_scene(test_tracks)
    scene_tracks = test_tracks[test_tracks["scene"] == scene_name].copy()
    start_frame = choose_start_frame(scene_tracks)
    initial_state = select_active_agents(state_at_frame(scene_tracks, start_frame), max_agents=18)
    initial_history = build_initial_history(scene_tracks, start_frame, initial_state, OBS_STEPS)
    tracked_ids = {agent["id"] for agent in initial_state}
    actual_state = [
        agent
        for agent in state_at_frame(scene_tracks, start_frame + HORIZON)
        if agent["id"] in tracked_ids
    ]

    prediction = smc_rollout_transformer(initial_history, model, HORIZON, seed=42)
    prediction["actual_t100"] = summarize_state(actual_state)
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
        "observation_steps": OBS_STEPS,
        "horizon_steps": HORIZON,
        "modeled_agents": len(initial_state),
    }
    prediction["model"] = model.metrics
    prediction["physics"] = {
        "body": "top-down capsule/circle approximation",
        "mass_kg": PERSON_MASS_KG,
        "collision_constraint": "hard projected non-overlap after every sampled transition",
        "importance_sampling": "proposal likelihood plus physical feasibility score",
        "boundary_constraint": "body centers are clamped inside image bounds",
    }

    summary_path = OUT_DIR / "summary.json"
    image_path = OUT_DIR / "rollout.png"
    model_path = OUT_DIR / "social_transformer_model.pt"
    summary_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": model.network.state_dict(),
            "feature_scaler": model.feature_scaler,
            "target_scaler": model.target_scaler,
            "residual_std": model.residual_std,
            "metrics": model.metrics,
            "obs_steps": model.obs_steps,
            "feature_dim": model.feature_dim,
            "feature_names": feature_names(),
        },
        model_path,
    )
    render_rollout(scene_tracks, start_frame, initial_state, prediction, actual_state, image_path)

    print(json.dumps({"summary": str(summary_path), "image": str(image_path), "model": str(model_path)}, indent=2))
    print(json.dumps(key_summary(prediction), indent=2))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        group["dt_prev"] = ((group["frame"] - group["prev_frame"]) / FPS).fillna(1.0 / FPS)
        group["dt_next"] = (group["next_frame"] - group["frame"]) / FPS
        group = group[(group["dt_next"] > 0) & (group["dt_next"] <= 1.5)].copy()
        group["vx"] = ((group["x"] - group["prev_x"]) / group["dt_prev"]).fillna(0)
        group["vy"] = ((group["y"] - group["prev_y"]) / group["dt_prev"]).fillna(0)
        group["dx"] = group["next_x"] - group["x"]
        group["dy"] = group["next_y"] - group["y"]
        groups.append(group)

    rows = pd.concat(groups, ignore_index=True)
    rows = add_social_features(rows)
    return rows.replace([np.inf, -np.inf], np.nan).dropna(subset=["dx", "dy"]).reset_index(drop=True)


def build_sequence_arrays(rows: pd.DataFrame, obs_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    rows = rows.reset_index(drop=True).copy()
    features = make_features(rows).astype(np.float32)
    targets = make_targets(rows).astype(np.float32)
    sequences: List[np.ndarray] = []
    sequence_targets: List[np.ndarray] = []

    for _, group in rows.groupby(["scene", "track_id"], sort=False):
        group = group.sort_values("frame")
        positions = group.index.to_numpy()
        if len(positions) == 0:
            continue
        for cursor, position in enumerate(positions):
            start = max(0, cursor - obs_steps + 1)
            window_positions = list(positions[start : cursor + 1])
            if len(window_positions) < obs_steps:
                window_positions = [window_positions[0]] * (obs_steps - len(window_positions)) + window_positions
            sequences.append(features[window_positions])
            sequence_targets.append(targets[position])

    return np.stack(sequences).astype(np.float32), np.stack(sequence_targets).astype(np.float32)


def train_transformer_model(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> TorchSequenceWorldModel:
    x_train, y_train = build_sequence_arrays(train_rows, OBS_STEPS)
    x_test, y_test = build_sequence_arrays(test_rows, OBS_STEPS)
    feature_dim = x_train.shape[-1]

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train_n = feature_scaler.fit_transform(x_train.reshape(-1, feature_dim)).reshape(x_train.shape)
    x_test_n = feature_scaler.transform(x_test.reshape(-1, feature_dim)).reshape(x_test.shape)
    y_train_n = target_scaler.fit_transform(y_train)
    y_test_n = target_scaler.transform(y_test)

    train_idx, val_idx = split_train_validation(len(x_train_n), seed=23)
    train_ds = TensorDataset(
        torch.tensor(x_train_n[train_idx], dtype=torch.float32),
        torch.tensor(y_train_n[train_idx], dtype=torch.float32),
    )
    val_x = torch.tensor(x_train_n[val_idx], dtype=torch.float32)
    val_y = torch.tensor(y_train_n[val_idx], dtype=torch.float32)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    network = SocialTransformer(feature_dim=feature_dim, obs_steps=OBS_STEPS)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.002, weight_decay=0.0008)
    loss_fn = nn.MSELoss()
    best_state = {name: tensor.detach().clone() for name, tensor in network.state_dict().items()}
    best_val = float("inf")
    patience_left = PATIENCE

    for epoch in range(1, MAX_EPOCHS + 1):
        network.train()
        running_loss = 0.0
        batch_count = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(network(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.5)
            optimizer.step()
            running_loss += float(loss.item())
            batch_count += 1

        network.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(network(val_x), val_y).item())
        print(
            f"epoch={epoch:02d} train_mse_scaled={running_loss / max(1, batch_count):.4f} "
            f"val_mse_scaled={val_loss:.4f}",
            flush=True,
        )
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_state = {name: tensor.detach().clone() for name, tensor in network.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    network.load_state_dict(best_state)
    network.eval()
    pred_train = target_scaler.inverse_transform(predict_batches(network, x_train_n))
    pred_test = target_scaler.inverse_transform(predict_batches(network, x_test_n))
    residual = y_train - pred_train
    residual_std = np.clip(residual.std(axis=0), 0.45, 18.0)
    rmse = np.sqrt(((y_test - pred_test) ** 2).mean(axis=0))
    ade = float(np.linalg.norm(y_test - pred_test, axis=1).mean())

    return TorchSequenceWorldModel(
        network=network,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        residual_std=residual_std,
        metrics={
            "model_type": "temporal social Transformer transition model",
            "architecture": "2-layer TransformerEncoder over 8-step social feature history",
            "train_sequences": int(len(x_train)),
            "test_sequences": int(len(x_test)),
            "dx_rmse_px": float(rmse[0]),
            "dy_rmse_px": float(rmse[1]),
            "ade_px": ade,
            "residual_dx_px": float(residual_std[0]),
            "residual_dy_px": float(residual_std[1]),
            "observed_collision_rate": estimate_collision_rate(test_rows),
            "epochs_trained": int(epoch),
            "best_validation_mse_scaled": float(best_val),
        },
        obs_steps=OBS_STEPS,
        feature_dim=feature_dim,
    )


def split_train_validation(size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(size)
    val_count = max(256, int(size * 0.12))
    return indexes[val_count:], indexes[:val_count]


def predict_batches(network: nn.Module, x_array: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x_array), batch_size):
            batch = torch.tensor(x_array[start : start + batch_size], dtype=torch.float32)
            predictions.append(network(batch).numpy())
    return np.concatenate(predictions, axis=0)


def build_initial_history(scene_tracks: pd.DataFrame, start_frame: int, initial_state: List[Dict], obs_steps: int) -> List[List[Dict]]:
    frames = list(range(start_frame - obs_steps + 1, start_frame + 1))
    initial_ids = [int(agent["id"]) for agent in initial_state]
    fallback = {int(agent["id"]): dict(agent) for agent in initial_state}
    history: List[List[Dict]] = []

    for frame in frames:
        by_id = {int(agent["id"]): agent for agent in state_at_frame(scene_tracks, frame)}
        step_state = []
        for agent_id in initial_ids:
            if agent_id in by_id:
                candidate = dict(by_id[agent_id])
                reference = fallback[agent_id]
                candidate["image_width"] = reference["image_width"]
                candidate["image_height"] = reference["image_height"]
                candidate["radius"] = reference["radius"]
                candidate["mass"] = reference["mass"]
                step_state.append(candidate)
            else:
                step_state.append(dict(fallback[agent_id]))
        history.append(step_state)

    return history


def smc_rollout_transformer(initial_history: List[List[Dict]], model: TorchSequenceWorldModel, horizon: int, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    particle_count = 72
    proposal_count = 3
    particles = [
        {
            "state": clone_state(initial_history[-1]),
            "history": clone_history(initial_history),
            "logp": 0.0,
            "reward": 0.0,
            "projection_cost": 0.0,
            "trace": compact_state(initial_history[-1]),
        }
        for _ in range(particle_count)
    ]
    ess_history = []

    for step in range(horizon):
        proposals = []
        bases = predict_deltas_for_histories([particle["history"] for particle in particles], model)
        for particle, base in zip(particles, bases):
            for _ in range(proposal_count):
                sampled_delta, proposal_logp = sample_proposal(base, model, rng)
                next_state, projection_cost = apply_deltas(particle["state"], sampled_delta)
                constraint_score = physics_score(next_state, projection_cost)
                next_history = (particle["history"] + [clone_state(next_state)])[-model.obs_steps :]
                proposals.append(
                    {
                        "state": next_state,
                        "history": next_history,
                        "logp": particle["logp"] + proposal_logp + constraint_score * 0.65,
                        "reward": particle["reward"] + constraint_score,
                        "projection_cost": particle["projection_cost"] + projection_cost,
                        "trace": append_trace(particle["trace"], next_state, step, horizon),
                    }
                )

        normalized = normalize_paths(proposals)
        ess = 1.0 / sum(max(1e-12, particle["probability"]) ** 2 for particle in normalized)
        ess_history.append(round(float(ess), 3))
        if ess < particle_count * 0.58:
            particles = systematic_resample_transformer(normalized, particle_count, rng)
        else:
            particles = normalized[:particle_count]
        if (step + 1) % 10 == 0 or step == 0:
            print(f"smc_step={step + 1:03d}/{horizon} ess={ess:.2f}", flush=True)

    weighted = normalize_paths(particles)
    return {
        "config": {
            "rollout": "Transformer-guided SMC importance sampling with systematic resampling",
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


def predict_deltas_from_history(history: List[List[Dict]], model: TorchSequenceWorldModel) -> np.ndarray:
    return predict_deltas_for_histories([history], model)[0]


def predict_deltas_for_histories(histories: List[List[List[Dict]]], model: TorchSequenceWorldModel) -> List[np.ndarray]:
    all_sequences = []
    sizes = []

    for history in histories:
        current_state = history[-1]
        feature_maps = []
        for step_state in history[-model.obs_steps :]:
            rows = state_to_feature_rows(step_state)
            features = make_features(rows)
            feature_maps.append({int(rows.iloc[index]["id"]): features[index] for index in range(len(rows))})

        for agent in current_state:
            agent_id = int(agent["id"])
            fallback = feature_maps[-1][agent_id]
            all_sequences.append([feature_map.get(agent_id, fallback) for feature_map in feature_maps])
        sizes.append(len(current_state))

    x = np.asarray(all_sequences, dtype=np.float32)
    x_n = model.feature_scaler.transform(x.reshape(-1, model.feature_dim)).reshape(x.shape)
    with torch.no_grad():
        scaled = model.network(torch.tensor(x_n, dtype=torch.float32)).numpy()
    predictions = model.target_scaler.inverse_transform(scaled)

    output = []
    cursor = 0
    for size in sizes:
        output.append(predictions[cursor : cursor + size])
        cursor += size
    return output


def sample_proposal(base: np.ndarray, model: TorchSequenceWorldModel, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    std = np.maximum(model.residual_std, 0.35)
    noise = rng.normal(0, std.reshape(1, 2), size=base.shape)
    z = noise / std.reshape(1, 2)
    proposal_logp = -0.5 * float(np.mean(z**2))
    return base + noise, proposal_logp


def systematic_resample_transformer(paths: List[Dict], count: int, rng: np.random.Generator) -> List[Dict]:
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
                "history": clone_history(path["history"]),
                "logp": 0.0,
                "probability": 1.0 / count,
                "reward": path["reward"],
                "projection_cost": path["projection_cost"],
                "trace": {agent_id: [list(point) for point in points] for agent_id, points in path["trace"].items()},
            }
        )
    return resampled


def clone_history(history: List[List[Dict]]) -> List[List[Dict]]:
    return [clone_state(step_state) for step_state in history]


if __name__ == "__main__":
    main()
