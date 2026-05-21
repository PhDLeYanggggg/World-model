from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.data_unification.splits import assign_scene_or_time_splits


def build_episodes_from_world_state(
    table: pd.DataFrame,
    dataset_name: str,
    output_root: str | Path = "data/stage5_episodes",
    past_horizon: int = 10,
    future_horizon: int = 100,
    minimum_agents: int = 1,
    max_agents: int = 128,
    quick: bool = True,
) -> Dict:
    table = assign_scene_or_time_splits(table)
    out_dir = Path(output_root) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    episode_id = 0
    for scene_id, scene_df in table.groupby("scene_id"):
        frames = sorted(scene_df["frame_id"].unique())
        stride = max(1, future_horizon // 2)
        for start in range(past_horizon - 1, len(frames) - future_horizon, stride):
            window = frames[start - past_horizon + 1 : start + future_horizon + 1]
            agents = set.intersection(*(set(scene_df.loc[scene_df["frame_id"] == frame, "agent_id"]) for frame in window))
            if len(agents) < minimum_agents:
                continue
            agents = sorted(agents)[:max_agents]
            rows = []
            for frame in window:
                frame_df = scene_df[(scene_df["frame_id"] == frame) & (scene_df["agent_id"].isin(agents))].sort_values("agent_id")
                rows.append(frame_df[["x", "y", "vx", "vy", "ax", "ay", "heading", "speed", "body_radius"]].to_numpy(dtype=np.float32))
            arr = np.stack(rows, axis=0)
            meta = {
                "episode_id": episode_id,
                "dataset_name": dataset_name,
                "scene_id": str(scene_id),
                "split": scene_df.loc[scene_df["frame_id"] == window[0], "split"].iloc[0],
                "past_horizon": past_horizon,
                "future_horizon": future_horizon,
                "can_evaluate_t100": future_horizon >= 100,
                "leakage_flags": [],
            }
            np.savez_compressed(out_dir / f"episode_{episode_id:05d}.npz", states=arr, meta=json.dumps(meta), agent_ids=np.asarray(agents, dtype=object))
            episodes.append(meta)
            episode_id += 1
            if quick and episode_id >= 32:
                break
        if quick and episode_id >= 32:
            break
    if len(set(ep["split"] for ep in episodes)) == 1 and len(episodes) >= 5:
        for idx, ep in enumerate(episodes):
            if idx < int(0.6 * len(episodes)):
                ep["split"] = "train"
            elif idx < int(0.8 * len(episodes)):
                ep["split"] = "val"
            else:
                ep["split"] = "test"
            path = out_dir / f"episode_{ep['episode_id']:05d}.npz"
            if path.exists():
                data = np.load(path, allow_pickle=True)
                np.savez_compressed(path, states=data["states"], meta=json.dumps(ep), agent_ids=data["agent_ids"])
    summary = {
        "dataset_name": dataset_name,
        "episodes": len(episodes),
        "t100_episodes": sum(1 for ep in episodes if ep["can_evaluate_t100"]),
        "train": sum(1 for ep in episodes if ep["split"] == "train"),
        "val": sum(1 for ep in episodes if ep["split"] == "val"),
        "test": sum(1 for ep in episodes if ep["split"] == "test"),
    }
    (out_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
