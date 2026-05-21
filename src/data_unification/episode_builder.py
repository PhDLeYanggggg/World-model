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
    for stale in out_dir.glob("episode_*.npz"):
        stale.unlink()
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
            candidates = sorted(agents)
            if max_agents == 1 and candidates:
                agents = [candidates[episode_id % len(candidates)]]
            else:
                agents = candidates[:max_agents]
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
                "coordinate_unit": str(scene_df["coordinate_unit"].iloc[0]) if "coordinate_unit" in scene_df else "unknown",
                "dt_s": float(scene_df["dt_s"].replace(0, np.nan).dropna().median()) if "dt_s" in scene_df and len(scene_df["dt_s"].dropna()) else 1.0,
                "official_velocity_source": str(scene_df["source_velocity_type"].iloc[0]) if "source_velocity_type" in scene_df else "causal_fd",
                "primary_agent_id": str(agents[0]),
                "start_frame": int(window[0]),
                "decision_frame": int(window[past_horizon - 1]),
                "end_frame": int(window[-1]),
                "frames": [int(frame) for frame in window],
                "leakage_flags": [],
            }
            np.savez_compressed(out_dir / f"episode_{episode_id:05d}.npz", states=arr, meta=json.dumps(meta), agent_ids=np.asarray(agents, dtype=object))
            episodes.append(meta)
            episode_id += 1
            if quick and episode_id >= 32:
                break
        if quick and episode_id >= 32:
            break
    split_counts = {split: sum(1 for ep in episodes if ep["split"] == split) for split in ["train", "val", "test"]}
    cross_split_primary_agent = has_cross_split_primary_agent(episodes)
    if (
        len(set(ep["split"] for ep in episodes)) == 1
        or any(count == 0 for count in split_counts.values())
        or cross_split_primary_agent
    ) and len(episodes) >= 5:
        unique_agents = sorted({str(ep.get("primary_agent_id", ep["episode_id"])) for ep in episodes})
        agent_to_split = {}
        if len(unique_agents) >= 3:
            n_train = max(1, int(round(0.6 * len(unique_agents))))
            n_val = max(1, int(round(0.2 * len(unique_agents))))
            if n_train + n_val >= len(unique_agents):
                n_train = max(1, len(unique_agents) - 2)
                n_val = 1
            for idx, agent in enumerate(unique_agents):
                if idx < n_train:
                    agent_to_split[agent] = "train"
                elif idx < n_train + n_val:
                    agent_to_split[agent] = "val"
                else:
                    agent_to_split[agent] = "test"
        for idx, ep in enumerate(episodes):
            agent = str(ep.get("primary_agent_id", ep["episode_id"]))
            if agent_to_split:
                ep["split"] = agent_to_split[agent]
            else:
                ep["split"] = "train" if idx < int(0.6 * len(episodes)) else ("val" if idx < int(0.8 * len(episodes)) else "test")
                ep.setdefault("leakage_flags", []).append("episode_index_split_used_due_to_too_few_unique_agents")
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


def stable_bucket(value: str) -> int:
    return sum(ord(ch) for ch in str(value)) % 100


def has_cross_split_primary_agent(episodes: list[Dict]) -> bool:
    agent_splits: Dict[str, set] = {}
    for ep in episodes:
        agent = str(ep.get("primary_agent_id", ep["episode_id"]))
        agent_splits.setdefault(agent, set()).add(ep.get("split", "unknown"))
    return any(len(splits) > 1 for splits in agent_splits.values())
