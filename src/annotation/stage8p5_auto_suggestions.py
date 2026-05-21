from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def scene_frame_split(df: pd.DataFrame) -> tuple[float, float]:
    frames = np.asarray(sorted(df["frame_id"].unique()), dtype=float)
    if len(frames) == 0:
        return 0.0, 0.0
    return float(np.quantile(frames, 0.6)), float(np.quantile(frames, 0.8))


def suggestions_from_world_state(dataset: str, scene_id: str, df: pd.DataFrame) -> Dict:
    train_end, _ = scene_frame_split(df)
    train = df[df["frame_id"] <= train_end].copy()
    if train.empty:
        train = df.copy()
    positions = train[["x", "y"]].to_numpy(dtype=float)
    pad = max(float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1])), 1.0) * 0.05
    xmin, ymin = positions.min(axis=0) - pad
    xmax, ymax = positions.max(axis=0) + pad
    boundary = [[float(xmin), float(ymin)], [float(xmax), float(ymin)], [float(xmax), float(ymax)], [float(xmin), float(ymax)]]
    endpoints = train.groupby("agent_id")[["x", "y"]].tail(1).to_numpy(dtype=float)
    goals = cluster_endpoints(endpoints)
    quality, reason = annotation_quality(dataset, goals, len(endpoints))
    return {
        "dataset_name": dataset,
        "scene_id": scene_id,
        "coordinate_unit": str(df["coordinate_unit"].iloc[0]) if "coordinate_unit" in df and len(df) else "unknown",
        "boundary_polygon": boundary,
        "goal_regions": goals,
        "annotation_quality": quality,
        "rule_confirmation_reason": reason,
        "train_endpoint_count": int(len(endpoints)),
        "test_endpoints_used": False,
    }


def cluster_endpoints(endpoints: np.ndarray, max_goals: int = 6) -> List[Dict]:
    if len(endpoints) == 0:
        return []
    centers = [endpoints[0]]
    spread = max(float(np.ptp(endpoints[:, 0])), float(np.ptp(endpoints[:, 1])), 1.0)
    min_sep = max(spread * 0.12, 2.0)
    for point in endpoints[1:]:
        d = np.linalg.norm(np.asarray(centers) - point[None, :], axis=1)
        if float(d.min()) > min_sep and len(centers) < max_goals:
            centers.append(point)
    assignments = []
    for point in endpoints:
        d = np.linalg.norm(np.asarray(centers) - point[None, :], axis=1)
        assignments.append(int(np.argmin(d)))
    goals = []
    for idx, center in enumerate(centers):
        support = sum(a == idx for a in assignments)
        goals.append(
            {
                "goal_id": f"silver_candidate_goal_{idx}",
                "region_type": "silver_goal_region",
                "center": [float(center[0]), float(center[1])],
                "radius": float(max(min_sep * 0.5, 1.0)),
                "support_count": int(support),
                "support_fraction": float(support / max(len(assignments), 1)),
                "confirmed_by_human": False,
                "confirmed_by_rule": True,
                "source": "train_split_endpoint_clustering",
            }
        )
    return goals


def annotation_quality(dataset: str, goals: List[Dict], endpoint_count: int) -> tuple[str, str]:
    pedestrian_like = dataset in {"trajnet", "eth_ucy", "sdd", "opentraj", "aerialmpt_long"}
    strong_support = endpoint_count >= 50 and len(goals) >= 2 and sum(g["support_count"] >= 3 for g in goals) >= 2
    if pedestrian_like and strong_support:
        return "silver", "high-confidence train-only endpoint clusters in a pedestrian/drone-like source"
    return "inferred_only", "insufficient support for silver rule confirmation"


def write_preview_png(dataset: str, scene_id: str, df: pd.DataFrame, suggestion: Dict, out_dir: str | Path = "outputs/figures/stage8p5_annotation_previews") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{dataset}_{scene_id}.png"
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        sample = df.sample(min(len(df), 5000), random_state=7) if len(df) > 5000 else df
        ax.scatter(sample["x"], sample["y"], s=1, alpha=0.25)
        for goal in suggestion["goal_regions"]:
            cx, cy = goal["center"]
            ax.scatter([cx], [cy], s=80, marker="x")
            ax.text(cx, cy, goal["goal_id"], fontsize=7)
        ax.set_title(f"{dataset}/{scene_id} {suggestion['annotation_quality']}")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return str(path)
    except Exception:  # noqa: BLE001
        return ""
