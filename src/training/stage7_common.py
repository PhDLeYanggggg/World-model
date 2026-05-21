from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.models.goal_intent_predictor import GoalIntentPredictor
from src.training.train_goal_intent_predictor import train_goal_predictor
from src.training.train_stage5b6_gated_residual import collect_examples


REPORT_DIR = Path("outputs/reports")
GOAL_CKPT = Path("outputs/checkpoints/stage7/goal_intent_predictor.json")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def ensure_goal_model() -> GoalIntentPredictor:
    if not GOAL_CKPT.exists():
        train_goal_predictor()
    return GoalIntentPredictor.load(GOAL_CKPT)


def goalbench_lookup() -> Dict[tuple, Dict]:
    records = load_json("data/goalbench/goalbench_records.json", [])
    return {(r["dataset"], int(r["episode_id"])): r for r in records}


def stage7_goal_features(goal_model: GoalIntentPredictor, record: Dict | None) -> np.ndarray:
    if not record:
        return np.zeros(8, dtype=np.float64)
    pred = goal_model.predict(record)
    probs = np.asarray(pred.get("probabilities", []), dtype=float)
    top = pred.get("top_goal_indices", [])
    distances = np.asarray(record.get("distances_to_goals", []), dtype=float)
    cosines = np.asarray(record.get("heading_cos_to_goals", []), dtype=float)
    top_idx = int(top[0]) if top else -1
    top_prob = float(probs[top_idx]) if top_idx >= 0 and top_idx < len(probs) else 0.0
    top3_mass = float(sum(probs[i] for i in top[:3] if i < len(probs))) if len(probs) else 0.0
    entropy = float(pred.get("entropy", 0.0))
    entropy_norm = entropy / np.log(max(len(probs), 2)) if len(probs) else 0.0
    top_dist = float(distances[top_idx]) if top_idx >= 0 and top_idx < len(distances) else 0.0
    dist_scale = max(float(np.nanpercentile(distances, 75)), 1.0) if len(distances) else 1.0
    top_cos = float(cosines[top_idx]) if top_idx >= 0 and top_idx < len(cosines) else 0.0
    goal_count = float(record.get("candidate_goal_count", 0))
    boundary = float(record.get("boundary_distance", 0.0))
    is_pixel = 1.0 if record.get("coordinate_unit") != "meter" else 0.0
    return np.asarray(
        [
            top_prob,
            top3_mass,
            entropy_norm,
            min(top_dist / dist_scale, 5.0),
            top_cos,
            min(goal_count / 6.0, 1.0),
            min(boundary, 20.0) / 20.0,
            is_pixel,
        ],
        dtype=np.float64,
    )


def collect_stage7_examples(split: str, feature_mode: str = "goal_scene_interaction") -> List[Dict]:
    baselines = load_json(REPORT_DIR / "stage5b_baseline_metrics.json", {"datasets": {}})
    base_rows = collect_examples(split, baselines)
    goal_model = ensure_goal_model()
    lookup = goalbench_lookup()
    rows = []
    for row in base_rows:
        grecord = lookup.get((row["dataset"], int(row["episode_id"])))
        gfeat = stage7_goal_features(goal_model, grecord)
        x = assemble_features(row["x"], gfeat, feature_mode)
        rows.append({**row, "x_stage7": x, "goal_features": gfeat, "goal_record": grecord})
    return rows


def assemble_features(base_x: np.ndarray, goal_features: np.ndarray, feature_mode: str) -> np.ndarray:
    base = np.asarray(base_x, dtype=np.float64)
    g = np.asarray(goal_features, dtype=np.float64)
    if feature_mode == "no_goal":
        return base
    if feature_mode == "goal_only":
        return np.concatenate([base[:18], g])
    if feature_mode == "scene_only":
        return np.concatenate([base[:18], g[[5, 6, 7]]])
    if feature_mode == "goal_scene":
        return np.concatenate([base[:18], g])
    if feature_mode == "goal_interaction":
        return np.concatenate([base, g[:6]])
    return np.concatenate([base, g])

