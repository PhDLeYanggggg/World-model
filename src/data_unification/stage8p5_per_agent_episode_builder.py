from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.scene.stage8p5_scene_gold_builder import load_stage8p5_scene_pack


FEATURE_COLUMNS = ["x", "y", "vx", "vy", "ax", "ay", "heading", "speed", "body_radius"]
OUT_DIR = Path("data/stage8p5_per_agent_episodes")
REPORT_DIR = Path("outputs/reports")


def available_stage8p5_world_sources(root: str | Path = "data/stage8p5_world_state") -> List[str]:
    base = Path(root)
    return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "world_state.csv").exists()) if base.exists() else []


def build_per_agent_episodes(datasets: List[str] | None = None, past_horizon: int = 10, future_horizon: int = 100, max_agents: int = 64, quick: bool = True) -> Dict:
    datasets = datasets or available_stage8p5_world_sources()
    summaries = []
    for dataset in datasets:
        summaries.append(build_dataset(dataset, past_horizon, future_horizon, max_agents, quick))
    payload = {"stage": "8.5", "datasets": summaries}
    write_report(payload)
    return payload


def build_dataset(dataset: str, past_horizon: int, future_horizon: int, max_agents: int, quick: bool) -> Dict:
    out_dir = OUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("episode_*.npz"):
        stale.unlink()
    df = pd.read_csv(Path("data/stage8p5_world_state") / dataset / "world_state.csv")
    df = df[df["valid"].astype(str).str.lower().isin(["true", "1"])] if "valid" in df else df
    df = df.sort_values(["scene_id", "frame_id", "agent_id"])
    metas = []
    episode_id = 0
    for scene_id, scene_df in df.groupby("scene_id"):
        frames = sorted(scene_df["frame_id"].unique().tolist())
        if len(frames) < past_horizon + 10:
            continue
        scene_future = min(future_horizon, max_future_for_scene(frames, past_horizon))
        if scene_future < 10:
            continue
        total = past_horizon + scene_future
        train_end_idx = int(0.6 * len(frames))
        val_end_idx = int(0.8 * len(frames))
        split_ranges = {
            "train": (0, max(0, train_end_idx - total)),
            "val": (train_end_idx, max(train_end_idx, val_end_idx - total)),
            "test": (val_end_idx, max(val_end_idx, len(frames) - total)),
        }
        step = max(1, total // 4)
        for split, (lo, hi) in split_ranges.items():
            if hi < lo:
                continue
            starts = list(range(lo, hi + 1, step))
            for start_idx in starts:
                window_frames = frames[start_idx : start_idx + total]
                if len(window_frames) < total:
                    continue
                sub = scene_df[scene_df["frame_id"].isin(window_frames)]
                agent_ids = rank_visible_agents(sub, window_frames, max_agents)
                if len(agent_ids) < 2:
                    continue
                states, mask = make_tensor(sub, window_frames, agent_ids)
                if mask[:past_horizon].sum(axis=0).max() < max(2, past_horizon // 2):
                    continue
                pack = load_stage8p5_scene_pack(dataset, str(scene_id))
                goal_candidates = pack.get("goal_regions", []) if pack else []
                labels = assign_per_agent_goals(states[-1, :, 0:2], goal_candidates, mask[-1])
                graph = neighbor_graph(states[past_horizon - 1], mask[past_horizon - 1])
                meta = {
                    "episode_id": episode_id,
                    "dataset_name": dataset,
                    "scene_id": str(scene_id),
                    "split": split,
                    "past_horizon": past_horizon,
                    "future_horizon": scene_future,
                    "official_eval_horizons": [h for h in [10, 25, 50, 100] if h <= scene_future],
                    "verified_t10": scene_future >= 10,
                    "verified_t25": scene_future >= 25,
                    "verified_t50": scene_future >= 50,
                    "verified_t100": scene_future >= 100,
                    "agent_ids": agent_ids,
                    "agent_count": len(agent_ids),
                    "dropped_agents": max(0, sub["agent_id"].astype(str).nunique() - len(agent_ids)),
                    "coordinate_unit": str(scene_df["coordinate_unit"].iloc[0]) if "coordinate_unit" in scene_df and len(scene_df) else "unknown",
                    "dt_s": float(np.nanmedian(scene_df["dt_s"].to_numpy(dtype=float))) if "dt_s" in scene_df and len(scene_df) else 1.0,
                    "annotation_quality": pack.get("annotation_quality", "not_available") if pack else "not_available",
                    "candidate_goal_source": "scene_annotation_train_only_or_confirmed",
                    "test_endpoints_used_for_goals": False,
                    "frame_start": int(window_frames[0]),
                    "frame_end": int(window_frames[-1]),
                    "hard_interaction": bool(min_pairwise_distance(states[:past_horizon], mask[:past_horizon]) < 5.0),
                    "baseline_failure_proxy": bool(path_curvature(states[:past_horizon, 0, :]) > 0.3),
                }
                np.savez_compressed(
                    out_dir / f"episode_{episode_id:05d}.npz",
                    states=states.astype(np.float32),
                    agent_mask=mask.astype(bool),
                    agent_ids=np.asarray(agent_ids, dtype=object),
                    per_agent_goal_labels=labels.astype(np.int32),
                    neighbor_graph=graph.astype(np.int32),
                    goal_candidates=json.dumps(goal_candidates),
                    scene_features=json.dumps({"annotation_quality": meta["annotation_quality"], "candidate_goal_count": len(goal_candidates)}),
                    meta=json.dumps(meta),
                )
                metas.append(meta)
                episode_id += 1
                if quick and episode_id >= 160:
                    break
            if quick and episode_id >= 160:
                break
        if quick and episode_id >= 160:
            break
    summary = summarize(dataset, metas)
    (out_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def max_future_for_scene(frames: List[int], past: int) -> int:
    available = max(0, len(frames) - past)
    for h in [100, 50, 25, 10]:
        if available >= h:
            return h
    return available


def rank_visible_agents(sub: pd.DataFrame, frames: List[int], max_agents: int) -> List[str]:
    counts = sub.groupby("agent_id")["frame_id"].nunique().sort_values(ascending=False)
    return [str(a) for a in counts.index[:max_agents]]


def make_tensor(sub: pd.DataFrame, frames: List[int], agents: List[str]) -> tuple[np.ndarray, np.ndarray]:
    frame_index = {f: i for i, f in enumerate(frames)}
    agent_index = {a: i for i, a in enumerate(agents)}
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


def assign_per_agent_goals(points: np.ndarray, goals: List[Dict], valid: np.ndarray) -> np.ndarray:
    labels = np.full(points.shape[0], -1, dtype=int)
    if not goals:
        return labels
    centers = np.asarray([g.get("center", [0.0, 0.0]) for g in goals], dtype=float)
    for i, point in enumerate(points):
        if not valid[i]:
            continue
        labels[i] = int(np.argmin(np.linalg.norm(centers - point[None, :], axis=1)))
    return labels


def neighbor_graph(frame: np.ndarray, valid: np.ndarray, k: int = 5) -> np.ndarray:
    idxs = np.where(valid)[0]
    graph = np.full((frame.shape[0], k), -1, dtype=int)
    if len(idxs) < 2:
        return graph
    pos = frame[idxs, 0:2]
    d = np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=2)
    d[d == 0] = np.inf
    for local, global_idx in enumerate(idxs):
        nbrs = np.argsort(d[local])[: min(k, len(idxs) - 1)]
        graph[int(global_idx), : len(nbrs)] = [int(idxs[n]) for n in nbrs]
    return graph


def min_pairwise_distance(states: np.ndarray, masks: np.ndarray) -> float:
    vals = []
    for frame, mask in zip(states, masks):
        pos = frame[mask, 0:2]
        if len(pos) < 2:
            continue
        d = np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=2)
        d[d == 0] = np.inf
        vals.append(float(np.min(d)))
    return min(vals) if vals else 999.0


def path_curvature(primary_hist: np.ndarray) -> float:
    if primary_hist.shape[0] < 3:
        return 0.0
    headings = np.arctan2(primary_hist[:, 3], primary_hist[:, 2])
    return float(np.sum(np.abs(np.angle(np.exp(1j * np.diff(headings)))))) if len(headings) else 0.0


def summarize(dataset: str, metas: List[Dict]) -> Dict:
    counts = [m["agent_count"] for m in metas]
    return {
        "dataset_name": dataset,
        "total_episodes": len(metas),
        "mean_agents_per_episode": float(np.mean(counts)) if counts else 0.0,
        "median_agents_per_episode": float(np.median(counts)) if counts else 0.0,
        "max_agents_per_episode": int(max(counts)) if counts else 0,
        "episodes_ge2_agents": sum(c >= 2 for c in counts),
        "episodes_ge5_agents": sum(c >= 5 for c in counts),
        "episodes_ge10_agents": sum(c >= 10 for c in counts),
        "gold_scene_episodes": sum(m["annotation_quality"] == "gold" for m in metas),
        "silver_scene_episodes": sum(m["annotation_quality"] == "silver" for m in metas),
        "inferred_only_episodes": sum(m["annotation_quality"] == "inferred_only" for m in metas),
        "verified_t10_episodes": sum(m["verified_t10"] for m in metas),
        "verified_t25_episodes": sum(m["verified_t25"] for m in metas),
        "verified_t50_episodes": sum(m["verified_t50"] for m in metas),
        "verified_t100_episodes": sum(m["verified_t100"] for m in metas),
        "pedestrian_drone_episodes": len(metas) if dataset in {"trajnet", "eth_ucy", "sdd", "opentraj", "aerialmpt_long"} else 0,
        "hard_interaction_episodes": sum(m["hard_interaction"] for m in metas),
        "baseline_failure_episodes": sum(m["baseline_failure_proxy"] for m in metas),
        "train_episodes": sum(m["split"] == "train" for m in metas),
        "val_episodes": sum(m["split"] == "val" for m in metas),
        "test_episodes": sum(m["split"] == "test" for m in metas),
    }


def write_report(payload: Dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage8p5_per_agent_episode_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    keys = [
        "dataset_name",
        "total_episodes",
        "mean_agents_per_episode",
        "episodes_ge2_agents",
        "episodes_ge5_agents",
        "episodes_ge10_agents",
        "silver_scene_episodes",
        "verified_t50_episodes",
        "verified_t100_episodes",
        "pedestrian_drone_episodes",
        "hard_interaction_episodes",
        "baseline_failure_episodes",
    ]
    lines = ["# Stage 8.5 Per-Agent Multi-Agent Episode Report", "", "| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in payload["datasets"]:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    (REPORT_DIR / "stage8p5_per_agent_episode_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
