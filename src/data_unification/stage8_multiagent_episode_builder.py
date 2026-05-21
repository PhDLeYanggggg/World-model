from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["x", "y", "vx", "vy", "ax", "ay", "heading", "speed", "body_radius"]
OUT_DIR = Path("data/stage8_multiagent_episodes")
REPORT_DIR = Path("outputs/reports")


def available_world_state_datasets(root: str | Path = "data/stage5b_world_state") -> List[str]:
    base = Path(root)
    return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "world_state.csv").exists()) if base.exists() else []


def build_stage8_multiagent_episodes(
    datasets: List[str] | None = None,
    past_horizon: int = 10,
    max_agents: int = 24,
    quick: bool = True,
) -> Dict:
    datasets = datasets or available_world_state_datasets()
    summaries = []
    for dataset in datasets:
        summaries.append(build_dataset(dataset, past_horizon=past_horizon, max_agents=max_agents, quick=quick))
    payload = {"stage": "8", "datasets": summaries}
    write_report(payload)
    return payload


def build_dataset(dataset: str, past_horizon: int = 10, max_agents: int = 24, quick: bool = True) -> Dict:
    out_dir = OUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("episode_*.npz"):
        stale.unlink()
    table = pd.read_csv(Path("data/stage5b_world_state") / dataset / "world_state.csv")
    table = table[table["valid"].astype(str).str.lower().isin(["true", "1"])]
    table = table.sort_values(["scene_id", "frame_id", "agent_id"])
    episodes = []
    episode_idx = 0
    for scene_id, scene_df in table.groupby("scene_id"):
        frames = sorted(scene_df["frame_id"].unique().tolist())
        if len(frames) < past_horizon + 2:
            continue
        max_future = min(100, len(frames) - past_horizon)
        if max_future >= 100:
            target = 100
        elif max_future >= 50:
            target = 50
        elif max_future >= 25:
            target = 25
        else:
            target = min(10, max_future)
        if target <= 0:
            continue
        total = past_horizon + target
        step = max(1, total // 2)
        for start in range(0, max(1, len(frames) - total + 1), step):
            window_frames = frames[start : start + total]
            if len(window_frames) < total:
                continue
            sub = scene_df[scene_df["frame_id"].isin(window_frames)]
            active_counts = sub.groupby("agent_id")["frame_id"].nunique()
            candidates = active_counts[active_counts >= max(2, int(0.7 * total))].index.astype(str).tolist()
            if len(candidates) < 2:
                continue
            candidates = rank_agents(sub, candidates, max_agents)
            states, mask = make_state_tensor(sub, window_frames, candidates)
            split = split_for_episode(episode_idx)
            meta = {
                "episode_id": episode_idx,
                "dataset_name": dataset,
                "scene_id": str(scene_id),
                "split": split,
                "past_horizon": past_horizon,
                "future_horizon": target,
                "can_evaluate_t50": target >= 50,
                "can_evaluate_t100": target >= 100,
                "coordinate_unit": str(scene_df["coordinate_unit"].iloc[0]) if "coordinate_unit" in scene_df else "unknown",
                "dt_s": float(np.nanmedian(scene_df["dt_s"].to_numpy(dtype=float))) if "dt_s" in scene_df else 1.0,
                "agent_ids": candidates,
                "agent_count": len(candidates),
                "dropped_agents": max(0, len(active_counts) - len(candidates)),
                "frame_start": int(window_frames[0]),
                "frame_end": int(window_frames[-1]),
                "source": "stage5b_world_state_multiagent_window",
            }
            np.savez_compressed(out_dir / f"episode_{episode_idx:05d}.npz", states=states.astype(np.float32), agent_mask=mask.astype(bool), meta=json.dumps(meta), agent_ids=np.asarray(candidates, dtype=object))
            episodes.append(meta)
            episode_idx += 1
            if quick and episode_idx >= 48:
                break
        if quick and episode_idx >= 48:
            break
    summary = summarize(dataset, episodes)
    (out_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def rank_agents(sub: pd.DataFrame, candidates: List[str], max_agents: int) -> List[str]:
    scores = []
    for agent in candidates:
        rows = sub[sub["agent_id"].astype(str) == str(agent)]
        speed = rows["speed"].astype(float).mean() if "speed" in rows else 0.0
        scores.append((str(agent), float(len(rows)), float(speed)))
    scores.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [item[0] for item in scores[:max_agents]]


def make_state_tensor(sub: pd.DataFrame, frames: List[int], agents: List[str]) -> tuple[np.ndarray, np.ndarray]:
    frame_index = {frame: idx for idx, frame in enumerate(frames)}
    agent_index = {agent: idx for idx, agent in enumerate(agents)}
    states = np.zeros((len(frames), len(agents), len(FEATURE_COLUMNS)), dtype=float)
    mask = np.zeros((len(frames), len(agents)), dtype=bool)
    for _, row in sub.iterrows():
        agent = str(row["agent_id"])
        if agent not in agent_index:
            continue
        t = frame_index.get(row["frame_id"])
        a = agent_index[agent]
        if t is None:
            continue
        for k, col in enumerate(FEATURE_COLUMNS):
            states[t, a, k] = float(row[col]) if col in row and pd.notna(row[col]) else 0.0
        mask[t, a] = True
    return states, mask


def split_for_episode(idx: int) -> str:
    mod = idx % 10
    if mod < 6:
        return "train"
    if mod < 8:
        return "val"
    return "test"


def summarize(dataset: str, episodes: List[Dict]) -> Dict:
    counts = [ep["agent_count"] for ep in episodes]
    summary = {
        "dataset_name": dataset,
        "total_episodes": len(episodes),
        "mean_agents_per_episode": float(np.mean(counts)) if counts else 0.0,
        "median_agents_per_episode": float(np.median(counts)) if counts else 0.0,
        "max_agents_per_episode": int(max(counts)) if counts else 0,
        "episodes_with_ge2_agents": sum(c >= 2 for c in counts),
        "episodes_with_ge5_agents": sum(c >= 5 for c in counts),
        "hard_interaction_episodes": 0,
        "verified_t50_episodes": sum(ep["can_evaluate_t50"] for ep in episodes),
        "verified_t100_episodes": sum(ep["can_evaluate_t100"] for ep in episodes),
        "pedestrian_drone_episodes": sum(dataset in {"trajnet", "eth_ucy"} for _ in episodes),
        "metric_episodes": sum(ep["coordinate_unit"] == "meter" for ep in episodes),
        "pixel_or_dataset_coordinate_episodes": sum(ep["coordinate_unit"] != "meter" for ep in episodes),
        "scene_gold_episodes": 0,
        "inferred_only_episodes": len(episodes),
        "train_episodes": sum(ep["split"] == "train" for ep in episodes),
        "val_episodes": sum(ep["split"] == "val" for ep in episodes),
        "test_episodes": sum(ep["split"] == "test" for ep in episodes),
    }
    return summary


def write_report(payload: Dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage8_multiagent_episode_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = payload["datasets"]
    keys = [
        "dataset_name",
        "total_episodes",
        "mean_agents_per_episode",
        "episodes_with_ge2_agents",
        "episodes_with_ge5_agents",
        "verified_t50_episodes",
        "verified_t100_episodes",
        "pedestrian_drone_episodes",
        "metric_episodes",
        "inferred_only_episodes",
    ]
    lines = ["# Stage 8 Multi-Agent Episode Report", "", "| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    (REPORT_DIR / "stage8_multiagent_episode_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

