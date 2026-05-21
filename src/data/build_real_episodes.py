from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.real_trajectory_loader import load_real_trajectory_table
from src.data.synthetic_physical_crowd import STATE_COLUMNS
from src.physics.scene_geometry import SceneSpec, scene_to_dict


REAL_HORIZONS = [10, 25, 50, 100]


def build_real_episodes(
    dataset: str,
    data_path: str | Path,
    output_root: str | Path = "data/real",
    quick: bool = False,
    history_steps: int = 6,
    max_agents: int = 30,
    max_episodes: int | None = None,
    velocity_source: str = "causal_fd",
) -> Dict:
    max_rows = 250_000 if quick else None
    table, source_meta = load_real_trajectory_table(dataset, data_path, quick=quick, max_rows=max_rows)
    summary = summarize_raw_table(table, source_meta)
    summary["velocity_source"] = velocity_source
    candidates_by_horizon = {h: count_windows(table, history_steps, h, min_agents=2) for h in REAL_HORIZONS}
    summary.update({f"samples_t{h}": int(candidates_by_horizon[h]) for h in REAL_HORIZONS})
    build_horizon = max([h for h in REAL_HORIZONS if candidates_by_horizon[h] > 0], default=0)
    summary["whether_t100_verified"] = bool(candidates_by_horizon[100] > 0)
    summary["build_horizon"] = int(build_horizon)
    summary["cannot_evaluate_t100"] = None if summary["whether_t100_verified"] else "No windows with at least two agents tracked through history+t100."
    output_root = Path(output_root) / dataset
    output_root.mkdir(parents=True, exist_ok=True)

    if build_horizon <= 0:
        summary.update({"train_episodes": 0, "val_episodes": 0, "test_episodes": 0, "mean_agents_per_episode": 0.0})
        write_summary(output_root, summary)
        return {"episodes": [], "summary": summary, "source_meta": source_meta}

    episodes = construct_windows(table, source_meta, history_steps, build_horizon, max_agents, max_episodes or (12 if quick else 120), velocity_source)
    split_episodes = assign_splits(episodes)
    for split in ["real_train", "real_val", "real_test"]:
        (output_root / split).mkdir(parents=True, exist_ok=True)
    for episode in split_episodes:
        split = episode["meta"]["split"]
        path = output_root / split / f"episode_{episode['meta']['episode_id']:04d}.npz"
        np.savez_compressed(
            path,
            states=episode["states"],
            meta=json.dumps(episode["meta"]),
            agent_ids=np.asarray(episode["agent_ids"], dtype=object),
            agent_types=np.asarray(episode["agent_types"], dtype=object),
            frame_times=episode.get("frame_times", np.asarray([], dtype=np.float32)),
            dt=np.asarray([episode.get("dt", episode["meta"].get("dt_seconds", 1.0))], dtype=np.float32),
        )
    train = [e for e in split_episodes if e["meta"]["split"] == "real_train"]
    val = [e for e in split_episodes if e["meta"]["split"] == "real_val"]
    test = [e for e in split_episodes if e["meta"]["split"] == "real_test"]
    summary.update(
        {
            "train_episodes": len(train),
            "val_episodes": len(val),
            "test_episodes": len(test),
            "mean_agents_per_episode": round(float(np.mean([e["states"].shape[1] for e in split_episodes])), 3) if split_episodes else 0.0,
            "split_policy": "scene split when possible; single-scene datasets use chronological non-overlapping windows",
        }
    )
    write_summary(output_root, summary)
    return {"episodes": split_episodes, "summary": summary, "source_meta": source_meta}


def summarize_raw_table(table: pd.DataFrame, source_meta: Dict) -> Dict:
    track_lengths = table.groupby(["scene_id", "agent_id"])["frame_id"].nunique()
    return {
        "dataset_name": source_meta.get("dataset_name", "unknown"),
        "total_scenes": int(table["scene_id"].nunique()),
        "total_agents": int(table["agent_id"].nunique()),
        "total_tracks": int(track_lengths.shape[0]),
        "total_frames": int(table["frame_id"].nunique()),
        "mean_track_length": round(float(track_lengths.mean()), 3) if len(track_lengths) else 0.0,
        "coordinate_unit": source_meta.get("coordinate_unit", "unknown"),
        "whether_metric_coordinates": bool(source_meta.get("whether_metric_coordinates", False)),
        "whether_scene_geometry_available": bool(source_meta.get("whether_scene_geometry_available", False)),
    }


def count_windows(table: pd.DataFrame, history_steps: int, horizon: int, min_agents: int = 2) -> int:
    count = 0
    window = history_steps + horizon
    for _, scene_df in table.groupby("scene_id"):
        frames = sorted(scene_df["frame_id"].unique())
        if len(frames) < window:
            continue
        frame_to_agents = {f: set(scene_df.loc[scene_df["frame_id"] == f, "agent_id"]) for f in frames}
        for start_index in range(history_steps - 1, len(frames) - horizon, max(1, horizon // 4)):
            needed = frames[start_index - history_steps + 1 : start_index + horizon + 1]
            common = set.intersection(*(frame_to_agents[f] for f in needed))
            if len(common) >= min_agents:
                count += 1
    return count


def construct_windows(table: pd.DataFrame, source_meta: Dict, history_steps: int, horizon: int, max_agents: int, max_episodes: int, velocity_source: str) -> List[Dict]:
    episodes = []
    episode_id = 0
    for scene_id, scene_df in table.groupby("scene_id"):
        scene_df = scene_df.sort_values(["frame_id", "agent_id"])
        frames = sorted(scene_df["frame_id"].unique())
        if len(frames) < history_steps + horizon:
            continue
        frame_to_agents = {f: set(scene_df.loc[scene_df["frame_id"] == f, "agent_id"]) for f in frames}
        stride = max(1, horizon // 2)
        for start_index in range(history_steps - 1, len(frames) - horizon, stride):
            needed = frames[start_index - history_steps + 1 : start_index + horizon + 1]
            common = sorted(set.intersection(*(frame_to_agents[f] for f in needed)))
            if len(common) < 2:
                continue
            common = common[:max_agents]
            episode = make_episode(scene_id, scene_df, needed, common, horizon, episode_id, source_meta, velocity_source)
            episodes.append(episode)
            episode_id += 1
            if len(episodes) >= max_episodes:
                return episodes
    return episodes


def make_episode(scene_id: str, scene_df: pd.DataFrame, frames: List[int], agent_ids: List[str], horizon: int, episode_id: int, source_meta: Dict, velocity_source: str = "causal_fd") -> Dict:
    bounds = scene_bounds(scene_df)
    origin = np.asarray([bounds["x_min"], bounds["y_min"]], dtype=np.float32)
    states = np.zeros((len(frames), len(agent_ids), len(STATE_COLUMNS)), dtype=np.float32)
    lookup = scene_df.set_index(["frame_id", "agent_id"])
    frame_times = frame_time_values(scene_df, frames)
    dt_seconds = float(np.median(np.diff(frame_times))) if len(frame_times) > 1 and np.all(np.isfinite(frame_times)) else 1.0
    last_history_frame = frames[5] if len(frames) > 5 else frames[0]
    goals = {}
    for agent_id in agent_ids:
        row = lookup.loc[(last_history_frame, agent_id)]
        vx, vy, _, _ = row_dynamics(row, velocity_source)
        direction = np.asarray([vx, vy], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-5:
            direction = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm
        goals[agent_id] = np.asarray([float(row["x"]), float(row["y"])], dtype=np.float32) - origin + direction * 20.0
    for t, frame in enumerate(frames):
        for i, agent_id in enumerate(agent_ids):
            row = lookup.loc[(frame, agent_id)]
            vx, vy, ax, ay = row_dynamics(row, velocity_source)
            speed = float(np.linalg.norm([vx, vy]))
            accel = float(np.linalg.norm([ax, ay]))
            radius = 0.35 if str(row["agent_type"]).lower() not in {"pedestrian", "person"} else 0.30
            states[t, i] = [
                row["x"],
                row["y"],
                vx,
                vy,
                ax,
                ay,
                float(np.arctan2(vy, vx)),
                radius,
                max(1.0, min(4.0, speed + 1.0)),
                max(1.5, min(5.0, accel + 2.0)),
                goals[agent_id][0],
                goals[agent_id][1],
                0,
                1,
                0,
                0,
                0,
                0,
            ]
            states[t, i, 0:2] -= origin
    meta = {
        "episode_id": int(episode_id),
        "scene_id": str(scene_id),
        "split": "unassigned",
        "dataset_name": source_meta.get("dataset_name", "real"),
        "frames": int(len(frames)),
        "agents": int(len(agent_ids)),
        "horizon": int(horizon),
        "state_columns": STATE_COLUMNS,
        "coordinate_unit": source_meta.get("coordinate_unit", "unknown"),
        "velocity_source": velocity_source,
        "dt_seconds": dt_seconds,
        "dt_source": "dataset_time" if "time" in scene_df.columns else "dense_frame_id",
        "whether_metric_coordinates": bool(source_meta.get("whether_metric_coordinates", False)),
        "whether_scene_geometry_available": bool(source_meta.get("whether_scene_geometry_available", False)),
        "scene_bounds": bounds,
        "scene_origin_world": {"x": float(origin[0]), "y": float(origin[1])},
        "true_t100_available": bool(horizon >= 100),
        "scene_info": {
            "walkable_area": "unknown",
            "obstacle_polygons": "unknown",
            "exit_regions": "unknown",
            "scene_image": None,
            "homography": None,
            "scale_m_per_px": None,
        },
    }
    return {
        "meta": meta,
        "states": states,
        "agent_ids": agent_ids,
        "agent_types": [str(lookup.loc[(frames[0], a)]["agent_type"]) for a in agent_ids],
        "scene": scene_from_bounds(scene_id, bounds, source_meta),
        "frame_times": frame_times.astype(np.float32),
        "dt": dt_seconds,
    }


def row_dynamics(row: pd.Series, velocity_source: str) -> Tuple[float, float, float, float]:
    source = velocity_source.lower()
    if source in {"native", "native_velocity"} and {"native_vx", "native_vy"}.issubset(row.index):
        return float(row["native_vx"]), float(row["native_vy"]), float(row.get("native_ax", row.get("ax", 0.0))), float(row.get("native_ay", row.get("ay", 0.0)))
    if source in {"central", "central_fd", "central_fd_diagnostic"} and {"central_vx", "central_vy"}.issubset(row.index):
        return float(row["central_vx"]), float(row["central_vy"]), float(row.get("central_ax", row.get("ax", 0.0))), float(row.get("central_ay", row.get("ay", 0.0)))
    if {"causal_vx", "causal_vy"}.issubset(row.index):
        return float(row["causal_vx"]), float(row["causal_vy"]), float(row.get("causal_ax", row.get("ax", 0.0))), float(row.get("causal_ay", row.get("ay", 0.0)))
    return float(row["vx"]), float(row["vy"]), float(row.get("ax", 0.0)), float(row.get("ay", 0.0))


def frame_time_values(scene_df: pd.DataFrame, frames: List[int]) -> np.ndarray:
    if "time" not in scene_df.columns:
        return np.asarray(frames, dtype=np.float32)
    mapping = scene_df.groupby("frame_id")["time"].median().to_dict()
    return np.asarray([mapping.get(frame, float(frame)) for frame in frames], dtype=np.float32)


def assign_splits(episodes: List[Dict]) -> List[Dict]:
    if not episodes:
        return []
    by_scene: Dict[str, List[Dict]] = {}
    for episode in episodes:
        by_scene.setdefault(episode["meta"]["scene_id"], []).append(episode)
    scenes = sorted(by_scene)
    if len(scenes) >= 3:
        train_scenes = set(scenes[: max(1, int(0.6 * len(scenes)))])
        val_scenes = set(scenes[max(1, int(0.6 * len(scenes))) : max(2, int(0.8 * len(scenes)))])
        for episode in episodes:
            scene = episode["meta"]["scene_id"]
            episode["meta"]["split"] = "real_train" if scene in train_scenes else ("real_val" if scene in val_scenes else "real_test")
    else:
        ordered = sorted(episodes, key=lambda e: e["meta"]["episode_id"])
        n = len(ordered)
        for idx, episode in enumerate(ordered):
            if idx < int(0.6 * n):
                episode["meta"]["split"] = "real_train"
            elif idx < int(0.8 * n):
                episode["meta"]["split"] = "real_val"
            else:
                episode["meta"]["split"] = "real_test"
    return episodes


def scene_bounds(scene_df: pd.DataFrame) -> Dict:
    pad = 2.0
    return {
        "x_min": float(scene_df["x"].min() - pad),
        "x_max": float(scene_df["x"].max() + pad),
        "y_min": float(scene_df["y"].min() - pad),
        "y_max": float(scene_df["y"].max() + pad),
    }


def scene_from_bounds(scene_id: str, bounds: Dict, source_meta: Dict | None = None) -> SceneSpec:
    width = max(1.0, bounds["x_max"] - bounds["x_min"])
    height = max(1.0, bounds["y_max"] - bounds["y_min"])
    has_geometry = bool((source_meta or {}).get("whether_scene_geometry_available", False))
    return SceneSpec(str(scene_id), width, height, [], {}, [], "real_unknown", has_geometry, has_geometry, has_geometry)


def write_summary(output_root: Path, summary: Dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "real_data_episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
