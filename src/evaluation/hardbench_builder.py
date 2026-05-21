from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from src.evaluation.baseline_benchmark_stage5b import load_dataset_episodes, rollout
from src.models.encoders.stage5b6_interaction_encoder import Stage5B6InteractionEncoder


REPORT_DIR = Path("outputs/reports")
OUT_DIR = Path("data/hardbench_v1")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def strongest_baselines() -> Dict:
    return load_json(REPORT_DIR / "stage5b_baseline_metrics.json", {"datasets": {}}).get("datasets", {})


def event_labels(scores: Dict) -> List[str]:
    events = []
    if scores["curvature"] > 0.08 or scores["heading_change"] > 0.7:
        events.append("turning")
    if scores["speed_change"] > 1.0:
        events.append("acceleration_change")
    if scores["deceleration_change"] > 0.5:
        events.append("deceleration_change")
    if scores["stop_duration"] > 0:
        events.append("stop_go")
    if scores["nearest_neighbor_distance"] < 2.0:
        events.append("close_interaction")
    if scores["time_to_collision"] < 3.0:
        events.append("near_collision")
    if scores["local_density"] > 0.2:
        events.append("high_density")
    if scores["crossing_angle"] > 0.7:
        events.append("crossing_paths")
    if scores["nonlinearity_score"] > 0.25:
        events.append("route_change")
    if scores["baseline_failure_score"] > 1.0:
        events.append("baseline_failure")
    return events or ["smooth_or_easy"]


def hardness(scores: Dict) -> str:
    score = (
        min(scores["heading_change"] / 2.0, 1.0)
        + min(scores["speed_change"] / 5.0, 1.0)
        + min(scores["acceleration_peak"] / 4.0, 1.0)
        + min(scores["jerk"] / 10.0, 1.0)
        + min(scores["baseline_failure_score"] / 5.0, 1.0)
        + min(scores["interaction_score"], 1.0)
        + min(scores["nonlinearity_score"] / 0.5, 1.0)
    ) / 7.0
    if score >= 0.66:
        return "extreme"
    if score >= 0.38:
        return "hard"
    if score >= 0.2:
        return "medium"
    return "easy"


def episode_scores(dataset: str, ep: Dict, baseline_name: str, encoder: Stage5B6InteractionEncoder) -> Dict:
    states = ep["states"]
    meta = ep["meta"]
    past = int(meta.get("past_horizon", 10))
    future_len = states.shape[0] - past
    dt = float(meta.get("dt_s", 1.0))
    traj = states[:, 0, :]
    vel = traj[:, 2:4]
    speed = np.linalg.norm(vel, axis=1)
    heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    xy = traj[:, 0:2]
    path_len = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0
    displacement = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) else 0.0
    accel = np.linalg.norm(traj[:, 4:6], axis=1)
    jerk = np.diff(accel) / max(dt, 1e-6) if len(accel) > 1 else np.asarray([0.0])
    horizon = 100 if future_len >= 100 else future_len
    base = rollout(states[:past], horizon, dt, baseline_name)[1:]
    true = states[past : past + horizon]
    err = np.linalg.norm(base[:, :, 0:2] - true[:, :, 0:2], axis=2)
    inter = encoder.encode_episode(meta)
    scores = {
        "curvature": float(np.mean(np.abs(np.diff(heading)))) if len(heading) > 1 else 0.0,
        "heading_change": float(np.sum(np.abs(np.diff(heading)))) if len(heading) > 1 else 0.0,
        "speed_change": float(np.sum(np.abs(np.diff(speed)))) if len(speed) > 1 else 0.0,
        "deceleration_change": float(np.sum(np.maximum(0.0, -np.diff(speed)))) if len(speed) > 1 else 0.0,
        "acceleration_peak": float(np.max(accel)) if len(accel) else 0.0,
        "jerk": float(np.mean(np.abs(jerk))) if len(jerk) else 0.0,
        "nearest_neighbor_distance": inter.nearest_neighbor_distance_min,
        "time_to_collision": inter.time_to_collision_min,
        "closing_speed": inter.closing_speed_max,
        "local_density": inter.local_density_mean,
        "crossing_angle": inter.crossing_angle_mean,
        "baseline_FDE": float(err[-1].mean()) if err.size else 0.0,
        "baseline_ADE": float(err.mean()) if err.size else 0.0,
        "baseline_failure_score": float(err[-1].mean()) if err.size else 0.0,
        "interaction_score": float(max(0.0, 1.0 - min(inter.nearest_neighbor_distance_min, 10.0) / 10.0) + inter.local_density_mean + min(max(inter.closing_speed_max, 0.0), 5.0) / 5.0) / 3.0,
        "nonlinearity_score": float(max(0.0, path_len - displacement) / max(path_len, 1e-6)),
        "stop_duration": float(np.sum(speed < max(0.05, np.percentile(speed, 10))) * dt) if len(speed) else 0.0,
    }
    events = event_labels(scores)
    level = hardness(scores)
    return {
        "dataset": dataset,
        "episode_id": int(meta.get("episode_id", -1)),
        "scene_id": meta.get("scene_id"),
        "split": meta.get("split"),
        "domain": "traffic" if dataset.startswith("tgsim") else "pedestrian",
        "future_horizon": int(future_len),
        "verified_t50": bool(future_len >= 50),
        "verified_t100": bool(future_len >= 100),
        "hardness": level,
        "events": events,
        "baseline": baseline_name,
        **{k: round(v, 6) if isinstance(v, float) else v for k, v in scores.items()},
    }


def build_hardbench(datasets: List[str] | None = None) -> Dict:
    baselines = strongest_baselines()
    datasets = datasets or sorted(baselines)
    records = []
    for dataset in datasets:
        if dataset not in baselines:
            continue
        baseline_name = baselines[dataset]["strongest_causal_baseline"]
        encoder = Stage5B6InteractionEncoder(dataset)
        for split in ["train", "val", "test"]:
            for ep in load_dataset_episodes(dataset, split=split):
                records.append(episode_scores(dataset, ep, baseline_name, encoder))
    apply_relative_hardness(records)
    return summarize(records)


def apply_relative_hardness(records: List[Dict]) -> None:
    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)
    for rows in by_dataset.values():
        scores = np.asarray([r["baseline_failure_score"] + r["interaction_score"] + r["nonlinearity_score"] for r in rows], dtype=float)
        if not len(scores):
            continue
        hard_cut = float(np.quantile(scores, 0.75))
        med_cut = float(np.quantile(scores, 0.45))
        for r, score in zip(rows, scores):
            if score >= hard_cut and r["hardness"] in {"easy", "medium"}:
                r["hardness"] = "hard"
            elif score >= med_cut and r["hardness"] == "easy":
                r["hardness"] = "medium"


def reliability(total_hard: int) -> str:
    if total_hard >= 100:
        return "strong"
    if total_hard >= 50:
        return "official"
    if total_hard >= 30:
        return "weak_evidence"
    return "diagnostic_only"


def summarize(records: List[Dict]) -> Dict:
    hard_records = [r for r in records if r["hardness"] in {"hard", "extreme"}]
    event_counts = Counter(event for r in hard_records for event in r["events"])
    dataset_counts = Counter(r["dataset"] for r in hard_records)
    horizon_counts = {
        "t10": sum(r["future_horizon"] >= 10 for r in hard_records),
        "t25": sum(r["future_horizon"] >= 25 for r in hard_records),
        "t50": sum(r["future_horizon"] >= 50 for r in hard_records),
        "t100": sum(r["future_horizon"] >= 100 for r in hard_records),
    }
    summary = {
        "total_episodes": len(records),
        "total_hard_episodes": len(hard_records),
        "hard_episodes_by_dataset": dict(dataset_counts),
        "hard_episodes_by_event_type": dict(event_counts),
        "hard_episodes_by_horizon": horizon_counts,
        "pedestrian_drone_hard_episodes": sum(r["domain"] == "pedestrian" for r in hard_records),
        "traffic_hard_episodes": sum(r["domain"] == "traffic" for r in hard_records),
        "verified_t50_hard_episodes": horizon_counts["t50"],
        "verified_t100_hard_episodes": horizon_counts["t100"],
        "gate_eligibility": reliability(len(hard_records)),
        "records": records,
    }
    return summary


def write_outputs(payload: Dict) -> Dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hardbench_v1_records.json").write_text(json.dumps(payload["records"], indent=2), encoding="utf-8")
    summary = {k: v for k, v in payload.items() if k != "records"}
    (OUT_DIR / "hardbench_v1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "hardbench_v1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = [
        {"field": "total_hard_episodes", "value": summary["total_hard_episodes"]},
        {"field": "pedestrian_drone_hard_episodes", "value": summary["pedestrian_drone_hard_episodes"]},
        {"field": "traffic_hard_episodes", "value": summary["traffic_hard_episodes"]},
        {"field": "verified_t50_hard_episodes", "value": summary["verified_t50_hard_episodes"]},
        {"field": "verified_t100_hard_episodes", "value": summary["verified_t100_hard_episodes"]},
        {"field": "gate_eligibility", "value": summary["gate_eligibility"]},
    ]
    text = "# HardBench-v1 Summary\n\n" + markdown_table(rows) + "\n## Hard Episodes By Dataset\n\n" + markdown_table([{"dataset": k, "hard": v} for k, v in summary["hard_episodes_by_dataset"].items()])
    (REPORT_DIR / "hardbench_v1_summary.md").write_text(text, encoding="utf-8")
    return payload


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"

