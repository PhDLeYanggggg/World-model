from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_world_state(dataset: str) -> pd.DataFrame:
    return pd.read_csv(Path("data/stage5b_world_state") / dataset / "world_state.csv")


def episode_paths(dataset: str):
    return sorted((Path("data/stage5b_episodes") / dataset).glob("episode_*.npz"))


def nearest_neighbor_features(table: pd.DataFrame, scene_id, frame_id, agent_id) -> Dict[str, float]:
    frame = table[(table["scene_id"].astype(str) == str(scene_id)) & (table["frame_id"].astype(int) == int(frame_id))]
    own = frame[frame["agent_id"].astype(str) == str(agent_id)]
    others = frame[frame["agent_id"].astype(str) != str(agent_id)]
    if own.empty or others.empty:
        return {"nearest_neighbor_distance_min": 999.0, "time_to_collision_min": 999.0, "interaction_density": 0.0, "closing_speed": 0.0}
    p = own[["x", "y"]].iloc[0].to_numpy(float)
    v = own[["vx", "vy"]].iloc[0].to_numpy(float)
    op = others[["x", "y"]].to_numpy(float)
    ov = others[["vx", "vy"]].to_numpy(float)
    rel = op - p
    dist = np.linalg.norm(rel, axis=1)
    relv = ov - v
    closing = -np.sum(rel * relv, axis=1) / np.maximum(dist, 1e-6)
    ttc = dist / np.maximum(closing, 1e-6)
    ttc[closing <= 0] = 999.0
    return {
        "nearest_neighbor_distance_min": float(np.min(dist)),
        "time_to_collision_min": float(np.min(ttc)),
        "interaction_density": float(np.mean(dist < 5.0)),
        "closing_speed": float(np.max(closing)),
    }


def classify_events(score: Dict[str, float]) -> List[str]:
    events = []
    if score["heading_change_total"] > 0.7 or score["mean_curvature"] > 0.08:
        events.append("turning")
    if score["speed_change_total"] > 0.8 or score["acceleration_peak"] > 1.5:
        events.append("acceleration_or_deceleration")
    if score["stop_duration"] > 0:
        events.append("stop_and_go")
    if score["nearest_neighbor_distance_min"] < 2.0:
        events.append("close_interaction")
    if score["time_to_collision_min"] < 3.0:
        events.append("near_collision")
    if score["interaction_density"] > 0.2:
        events.append("high_density")
    if score["trajectory_non_linearity"] > 0.25:
        events.append("long_horizon_non_linear_motion")
    if not events:
        events.append("smooth_or_easy")
    return events


def hardness_level(hard_score: float) -> str:
    if hard_score >= 0.66:
        return "hard"
    if hard_score >= 0.33:
        return "medium"
    return "easy"
