from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from src.evaluation.interaction_event_miner import classify_events, episode_paths, hardness_level, load_world_state, nearest_neighbor_features
from src.evaluation.baseline_benchmark_stage5b import rollout


def mine_dataset(dataset: str) -> Dict:
    table = load_world_state(dataset)
    records = []
    for path in episode_paths(dataset):
        data = np.load(path, allow_pickle=True)
        states = data["states"].astype(np.float32)
        meta = json.loads(str(data["meta"].item()))
        past = int(meta.get("past_horizon", 10))
        future = states[past:]
        hist = states[:past]
        dt = float(meta.get("dt_s", 1.0))
        primary = meta.get("primary_agent_id", data["agent_ids"].tolist()[0])
        frame = meta.get("decision_frame", meta.get("frames", [0])[-1])
        kin = trajectory_scores(hist[:, 0, :], future[:, 0, :], dt)
        nn = nearest_neighbor_features(table, meta.get("scene_id"), frame, primary)
        proxy_cv = constant_error_proxy(hist, future, dt, "constant_velocity_causal_fd")
        proxy_turn = constant_error_proxy(hist, future, dt, "constant_turn_rate_velocity")
        score = {**kin, **nn, "constant_velocity_error_proxy": proxy_cv, "constant_turn_error_proxy": proxy_turn}
        score["hard_score"] = normalize_hard_score(score)
        score["hardness"] = hardness_level(score["hard_score"])
        score["events"] = classify_events(score)
        score["dataset_name"] = dataset
        score["episode_id"] = int(meta.get("episode_id", -1))
        score["split"] = meta.get("split", "unknown")
        score["can_evaluate_t100"] = bool(meta.get("can_evaluate_t100", False))
        score["future_horizon"] = int(meta.get("future_horizon", future.shape[0]))
        records.append(score)
    return summarize(dataset, records)


def trajectory_scores(hist: np.ndarray, future: np.ndarray, dt: float) -> Dict[str, float]:
    traj = np.concatenate([hist, future], axis=0)
    xy = traj[:, 0:2]
    vel = traj[:, 2:4]
    speed = np.linalg.norm(vel, axis=1)
    heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    dheading = np.diff(heading)
    accel = np.linalg.norm(traj[:, 4:6], axis=1)
    jerk = np.diff(accel) / max(dt, 1e-6) if len(accel) > 1 else np.asarray([0.0])
    displacement = np.linalg.norm(xy[-1] - xy[0])
    path_len = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    return {
        "mean_curvature": float(np.mean(np.abs(dheading))) if len(dheading) else 0.0,
        "max_curvature": float(np.max(np.abs(dheading))) if len(dheading) else 0.0,
        "heading_change_total": float(np.sum(np.abs(dheading))) if len(dheading) else 0.0,
        "speed_change_total": float(np.sum(np.abs(np.diff(speed)))) if len(speed) > 1 else 0.0,
        "acceleration_peak": float(np.max(accel)) if len(accel) else 0.0,
        "jerk_score": float(np.mean(np.abs(jerk))) if len(jerk) else 0.0,
        "trajectory_non_linearity": float(max(0.0, path_len - displacement) / max(path_len, 1e-6)),
        "stop_duration": float(np.sum(speed < max(0.05, np.percentile(speed, 10))) * dt),
        "route_deviation_score": float(max(0.0, path_len - displacement)),
        "crossing_angle": 0.0,
        "local_density": 0.0,
    }


def constant_error_proxy(hist: np.ndarray, future: np.ndarray, dt: float, model: str) -> float:
    if future.size == 0:
        return 0.0
    pred = rollout(hist, future.shape[0], dt, model)[1:]
    err = np.linalg.norm(pred[:, :, 0:2] - future[:, :, 0:2], axis=2)
    return float(err[-1].mean())


def normalize_hard_score(s: Dict[str, float]) -> float:
    vals = [
        min(s["heading_change_total"] / 2.0, 1.0),
        min(s["speed_change_total"] / 5.0, 1.0),
        min(s["acceleration_peak"] / 4.0, 1.0),
        min(s["jerk_score"] / 10.0, 1.0),
        min(s["trajectory_non_linearity"] / 0.5, 1.0),
        1.0 if s["nearest_neighbor_distance_min"] < 2.0 else 0.0,
        1.0 if s["time_to_collision_min"] < 3.0 else 0.0,
        min(s["constant_velocity_error_proxy"] / 5.0, 1.0),
    ]
    return float(np.mean(vals))


def summarize(dataset: str, records: List[Dict]) -> Dict:
    apply_relative_hardness(records)
    counts = Counter(r["hardness"] for r in records)
    events = Counter(e for r in records for e in r["events"])
    return {
        "dataset_name": dataset,
        "episodes": records,
        "total_episodes": len(records),
        "easy_episodes": counts["easy"],
        "medium_episodes": counts["medium"],
        "hard_episodes": counts["hard"],
        "hard_ratio": round(counts["hard"] / max(len(records), 1), 4),
        "t100_hard_episodes": sum(1 for r in records if r["hardness"] == "hard" and r["can_evaluate_t100"]),
        "t50_hard_episodes": sum(1 for r in records if r["hardness"] == "hard" and r["future_horizon"] >= 50),
        "t10_hard_episodes": sum(1 for r in records if r["hardness"] == "hard" and r["future_horizon"] >= 10),
        "dominant_hard_event_types": [event for event, _ in events.most_common(5)],
        "large_enough_for_training": counts["hard"] >= 5,
        "large_enough_for_evaluation": counts["hard"] >= 3,
        "hardness_policy": "absolute score plus dataset-relative top quantile; hard labels are for evaluation stratification, not semantic ground truth",
    }


def apply_relative_hardness(records: List[Dict]) -> None:
    if not records:
        return
    scores = np.asarray([r["hard_score"] for r in records], dtype=float)
    hard_cut = max(0.33, float(np.quantile(scores, 0.75)))
    med_cut = max(0.18, float(np.quantile(scores, 0.45)))
    for record in records:
        if record["hard_score"] >= hard_cut:
            record["hardness"] = "hard"
        elif record["hard_score"] >= med_cut:
            record["hardness"] = "medium"
        else:
            record["hardness"] = "easy"


def write_hard_subset_report(rows: Iterable[Dict], path: str | Path = "outputs/reports/stage5b5_hard_subset_summary.md") -> List[Dict]:
    summaries = list(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    lines = [
        "# Stage 5B.5 Hard Subset Summary",
        "",
        "Hard subset mining is used for evaluation stratification and training weights. It is not used as a future-derived input feature to the model.",
        "",
        "| dataset | total | easy | medium | hard | hard_ratio | t100_hard | t50_hard | t10_hard | events | train_ok | eval_ok |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset_name']} | {row['total_episodes']} | {row['easy_episodes']} | {row['medium_episodes']} | {row['hard_episodes']} | "
            f"{row['hard_ratio']} | {row['t100_hard_episodes']} | {row['t50_hard_episodes']} | {row['t10_hard_episodes']} | "
            f"{', '.join(row['dominant_hard_event_types'])} | {row['large_enough_for_training']} | {row['large_enough_for_evaluation']} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries
