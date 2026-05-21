from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def audit_world_state_table(dataset_name: str, table: pd.DataFrame) -> Dict:
    track_lengths = table.groupby("agent_id")["frame_id"].nunique() if len(table) else pd.Series(dtype=float)
    speed = table["speed"].to_numpy(dtype=float) if "speed" in table else np.asarray([])
    accel = table["acceleration_norm"].to_numpy(dtype=float) if "acceleration_norm" in table else np.asarray([])
    dt = table["dt_s"].to_numpy(dtype=float) if "dt_s" in table else np.asarray([])
    return {
        "dataset_name": dataset_name,
        "total_scenes": int(table["scene_id"].nunique()) if len(table) else 0,
        "total_agents": int(table["agent_id"].nunique()) if len(table) else 0,
        "total_tracks": int(table["agent_id"].nunique()) if len(table) else 0,
        "total_frames": int(table["frame_id"].nunique()) if len(table) else 0,
        "total_episodes": 0,
        "samples_t10": "pending_episode_build",
        "samples_t25": "pending_episode_build",
        "samples_t50": "pending_episode_build",
        "samples_t100": "pending_episode_build",
        "mean_track_length": float(track_lengths.mean()) if len(track_lengths) else 0.0,
        "p50_track_length": float(track_lengths.quantile(0.5)) if len(track_lengths) else 0.0,
        "p95_track_length": float(track_lengths.quantile(0.95)) if len(track_lengths) else 0.0,
        "frame_rate_distribution": "derived from dt",
        "dt_distribution": summarize(dt),
        "speed_distribution": summarize(speed),
        "acceleration_distribution": summarize(accel),
        "heading_rate_distribution": "pending",
        "missing_frame_rate": "pending",
        "abnormal_jump_count": "pending",
        "coordinate_outlier_count": "pending",
        "agent_type_distribution": table["agent_type"].value_counts().to_dict() if len(table) else {},
        "scene_type_distribution": {},
        "interaction_density": "pending",
        "collision_or_close_pass_rate": "pending",
        "map_availability_rate": 0.0,
        "t100_verified_rate": "pending",
    }


def summarize(values: np.ndarray) -> Dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "p50": None, "p95": None}
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


def write_data_quality_report(audits: list[Dict], path: str | Path = "outputs/reports/data_quality_audit_stage5.md") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Stage 5 Data Quality Audit", "", "| dataset | scenes | agents | frames | mean_track_length | speed | accel | agent_types |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in audits:
        lines.append(f"| {row['dataset_name']} | {row['total_scenes']} | {row['total_agents']} | {row['total_frames']} | {round(row['mean_track_length'], 3)} | {row['speed_distribution']} | {row['acceleration_distribution']} | {row['agent_type_distribution']} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
