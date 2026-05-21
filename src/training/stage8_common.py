from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.evaluation.baseline_benchmark_stage5b import rollout
from src.evaluation.baseline_failure_oracle import threshold_for
from src.evaluation.stage8_goalbench_gold import available_stage8_datasets, load_multiagent_episodes


REPORT_DIR = Path("outputs/reports")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def strongest_baselines() -> Dict:
    return load_json(REPORT_DIR / "stage5b_baseline_metrics.json", {"datasets": {}}).get("datasets", {})


def goalbench_lookup() -> Dict[tuple, Dict]:
    records = load_json("data/stage8_goalbench_gold/goalbench_gold_records.json", [])
    return {(r["dataset"], int(r["episode_id"])): r for r in records}


def collect_stage8_examples(split: str = "train") -> List[Dict]:
    baselines = strongest_baselines()
    gl = goalbench_lookup()
    rows = []
    for dataset in available_stage8_datasets():
        if dataset not in baselines:
            continue
        baseline_name = baselines[dataset]["strongest_causal_baseline"]
        for ep in load_multiagent_episodes(dataset, split=split):
            states = ep["states"]
            meta = ep["meta"]
            past = int(meta["past_horizon"])
            future = int(meta["future_horizon"])
            dt = float(meta.get("dt_s", 1.0))
            base = rollout(states[:past], future, dt, baseline_name)[1:]
            true = states[past : past + future]
            goal = gl.get((dataset, int(meta["episode_id"])))
            for h in [v for v in [1, 10, 25, 50, 100] if v <= future]:
                b_fde = float(np.linalg.norm(base[h - 1, :, 0:2] - true[h - 1, :, 0:2], axis=1).mean())
                rows.append(
                    {
                        "dataset": dataset,
                        "episode_id": int(meta["episode_id"]),
                        "split": split,
                        "horizon": h,
                        "future_len": future,
                        "baseline_name": baseline_name,
                        "baseline_fde": b_fde,
                        "baseline_failure": bool(b_fde > threshold_for(dataset, h, subset="hard")),
                        "hardness": "hard" if b_fde > threshold_for(dataset, h, subset="hard") else "easy",
                        "x": stage8_feature_vector(states[:past], meta, goal, h, future),
                        "y": (true[h - 1, 0, 0:2] - base[h - 1, 0, 0:2]).astype(np.float64),
                    }
                )
    return rows


def stage8_feature_vector(hist: np.ndarray, meta: Dict, goal: Dict | None, horizon: int, future: int) -> np.ndarray:
    primary = hist[-1, 0]
    prev = hist[-2, 0] if hist.shape[0] >= 2 else primary
    positions = hist[-1, :, 0:2]
    valid = np.linalg.norm(positions, axis=1) > 0
    active = positions[valid]
    if len(active) >= 2:
        d = np.linalg.norm(active[None, :, :] - active[:, None, :], axis=2)
        d[d == 0] = np.inf
        nn = float(np.min(d))
        density = float(len(active) / max(np.prod(np.maximum(active.max(axis=0) - active.min(axis=0), 1.0)), 1.0))
    else:
        nn = 999.0
        density = 0.0
    heading_delta = float(np.angle(np.exp(1j * (primary[6] - prev[6]))))
    goal_feats = goal_features(goal)
    base = np.asarray(
        [
            primary[2],
            primary[3],
            primary[4],
            primary[5],
            primary[7],
            np.sin(primary[6]),
            np.cos(primary[6]),
            heading_delta,
            float(primary[7] - prev[7]),
            min(float(meta.get("agent_count", 1)) / 24.0, 1.0),
            min(nn, 20.0) / 20.0,
            min(density, 1.0),
            horizon / max(future, 1),
            1.0 if meta.get("coordinate_unit") == "meter" else 0.0,
            1.0 if meta.get("dataset_name", "").startswith("tgsim") else 0.0,
        ],
        dtype=np.float64,
    )
    return np.concatenate([base, goal_feats])


def goal_features(goal: Dict | None) -> np.ndarray:
    if not goal or goal.get("candidate_goal_count", 0) <= 0:
        return np.zeros(8, dtype=np.float64)
    distances = np.asarray(goal.get("distances_to_goals", []), dtype=float)
    angles = np.asarray(goal.get("angle_to_goals", []), dtype=float)
    if len(distances) == 0:
        return np.zeros(8, dtype=np.float64)
    nearest = int(np.argmin(distances))
    dist_scale = max(float(np.nanpercentile(distances, 75)), 1.0)
    sorted_d = np.sort(distances)
    margin = float((sorted_d[1] - sorted_d[0]) / dist_scale) if len(sorted_d) > 1 else 1.0
    return np.asarray(
        [
            min(float(distances[nearest] / dist_scale), 5.0),
            float(angles[nearest]) if nearest < len(angles) else 0.0,
            min(float(goal.get("candidate_goal_count", 0)) / 8.0, 1.0),
            float(goal.get("agent_count", 1)) / 24.0,
            margin,
            1.0 if goal.get("goal_quality") in {"gold", "silver"} else 0.0,
            1.0 if goal.get("route_distance_available") else 0.0,
            1.0 if goal.get("goal_quality") == "inferred_only" else 0.0,
        ],
        dtype=np.float64,
    )

