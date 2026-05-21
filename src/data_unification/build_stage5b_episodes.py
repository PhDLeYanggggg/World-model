from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_unification.convert_to_world_state import convert_real_dataset
from src.data_unification.episode_builder import build_episodes_from_world_state


DATASET_DOMAINS = {
    "tgsim": "traffic",
    "tgsim_i90": "traffic",
    "tgsim_other": "traffic",
    "trajnet": "pedestrian",
    "eth_ucy": "pedestrian",
    "sdd": "drone",
    "opendd": "traffic",
    "ngsim": "traffic",
}


def build_stage5b_dataset(dataset: str, data_path: str, quick: bool = True) -> Dict:
    table, meta = convert_real_dataset(dataset, data_path, output_root="data/stage5b_world_state", quick=quick)
    summary = summarize_world_state(dataset, table, meta)
    build_horizon = 100 if summary["samples_t100"] > 0 else (50 if summary["samples_t50"] > 0 else (25 if summary["samples_t25"] > 0 else 10))
    episode_summary = build_episodes_from_world_state(
        table,
        dataset,
        output_root="data/stage5b_episodes",
        past_horizon=10,
        future_horizon=build_horizon,
        minimum_agents=1,
        max_agents=1,
        quick=quick,
    )
    summary.update(
        {
            "train_episodes": episode_summary["train"],
            "val_episodes": episode_summary["val"],
            "test_episodes": episode_summary["test"],
            "actual_verified_t100": bool(episode_summary["t100_episodes"] > 0),
            "official_eval_horizons": [h for h in [1, 10, 25, 50, 100] if h <= build_horizon],
            "split_policy": "scene split when possible; quick single-scene fallback uses primary-agent-disjoint split",
            "leakage_flags": [],
        }
    )
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"stage5b_episode_summary_{dataset}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_world_state(dataset: str, table: pd.DataFrame, meta: Dict) -> Dict:
    track_lengths = table.groupby(["scene_id", "agent_id"])["frame_id"].nunique()
    samples = {h: count_windows(table, h) for h in [10, 25, 50, 100]}
    return {
        "dataset_name": dataset,
        "domain": DATASET_DOMAINS.get(dataset, "unknown"),
        "total_scenes": int(table["scene_id"].nunique()),
        "total_tracks": int(track_lengths.shape[0]),
        "total_agents": int(table["agent_id"].nunique()),
        "total_frames": int(table["frame_id"].nunique()),
        "mean_track_length": round(float(track_lengths.mean()), 3) if len(track_lengths) else 0.0,
        "p50_track_length": round(float(track_lengths.quantile(0.5)), 3) if len(track_lengths) else 0.0,
        "p95_track_length": round(float(track_lengths.quantile(0.95)), 3) if len(track_lengths) else 0.0,
        "coordinate_unit": str(table["coordinate_unit"].iloc[0]) if len(table) else "unknown",
        "is_metric": bool(meta.get("whether_metric_coordinates", table["coordinate_unit"].iloc[0] == "meter" if len(table) else False)),
        "has_scene_map": bool(meta.get("whether_scene_geometry_available", False)),
        "has_homography": False,
        "has_agent_type": "agent_type" in table.columns,
        "has_heading": "heading" in table.columns,
        "has_velocity": "vx" in table.columns,
        "official_velocity_source": "causal_fd",
        "samples_t10": samples[10],
        "samples_t25": samples[25],
        "samples_t50": samples[50],
        "samples_t100": samples[100],
        "actual_verified_t100": samples[100] > 0,
    }


def count_windows(table: pd.DataFrame, horizon: int, past: int = 10) -> int:
    count = 0
    need = past + horizon
    for _, scene_df in table.groupby("scene_id"):
        frames = sorted(scene_df["frame_id"].unique())
        if len(frames) < need:
            continue
        frame_agents = {frame: set(scene_df.loc[scene_df["frame_id"] == frame, "agent_id"]) for frame in frames}
        stride = max(1, horizon // 2)
        for start in range(past - 1, len(frames) - horizon, stride):
            window = frames[start - past + 1 : start + horizon + 1]
            if set.intersection(*(frame_agents[frame] for frame in window)):
                count += 1
    return count


def write_all_summary(summaries: List[Dict]) -> None:
    out = Path("outputs/reports/stage5b_episode_summary_all.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        out.write_text("_No Stage 5B datasets converted._\n", encoding="utf-8")
        return
    keys = list(summaries[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in summaries:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
