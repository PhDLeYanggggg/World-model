from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.models.stage5b6_gated_residual_model import fit_logistic, fit_weighted_ridge
from src.models.stage7_interaction_auxiliary import Stage7InteractionAuxiliary
from src.training.stage7_common import collect_stage7_examples
from src.training.train_baseline_failure_predictor import auroc, auprc


REPORT_DIR = Path("outputs/reports")
CKPT = Path("outputs/checkpoints/stage7/interaction_auxiliary.json")


def train_interaction_auxiliary() -> Dict:
    train = collect_stage7_examples("train", "goal_scene_interaction")
    test = collect_stage7_examples("test", "goal_scene_interaction")
    if not train:
        return {"stage": "7", "available": False, "reason": "no training rows"}
    x = np.stack([r["x_stage7"] for r in train])
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    xz = (x - mean) / np.maximum(scale, 1e-6)
    targets = make_targets(train)
    heads = {
        "future_nearest_neighbor_distance": fit_weighted_ridge(xz, targets["future_nearest_neighbor_distance"][:, None], np.ones(len(train)), ridge=5e-2).ravel().tolist(),
        "future_ttc_min": fit_weighted_ridge(xz, targets["future_ttc_min"][:, None], np.ones(len(train)), ridge=5e-2).ravel().tolist(),
        "close_pass_event_prob": fit_logistic(xz, targets["close_pass_event"], np.ones(len(train)), steps=400, lr=0.08, l2=1e-3).tolist(),
        "density_increase_event_prob": fit_logistic(xz, targets["density_increase_event"], np.ones(len(train)), steps=400, lr=0.08, l2=1e-3).tolist(),
        "crossing_conflict_event_prob": fit_logistic(xz, targets["crossing_conflict_event"], np.ones(len(train)), steps=400, lr=0.08, l2=1e-3).tolist(),
        "stop_go_event_prob": fit_logistic(xz, targets["stop_go_event"], np.ones(len(train)), steps=400, lr=0.08, l2=1e-3).tolist(),
        "local_congestion_event_prob": fit_logistic(xz, targets["local_congestion_event"], np.ones(len(train)), steps=400, lr=0.08, l2=1e-3).tolist(),
    }
    model = Stage7InteractionAuxiliary({"model": "stage7_interaction_auxiliary", "x_mean": mean.tolist(), "x_scale": scale.tolist(), "heads": heads})
    model.save(CKPT)
    metrics = evaluate(model, test)
    payload = {
        "stage": "7",
        "checkpoint": str(CKPT),
        "train_samples": len(train),
        "test_samples": len(test),
        "metrics": metrics,
        "note": "Most converted episodes are single-primary-agent windows, so interaction labels remain weak diagnostics.",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage7_interaction_auxiliary_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORT_DIR / "stage7_interaction_auxiliary_report.md").write_text(report(payload), encoding="utf-8")
    return payload


def make_targets(rows: List[Dict]) -> Dict[str, np.ndarray]:
    nn = np.asarray([max(0.1, 10.0 * (1.0 - min(float(r["x"][18]) if len(r["x"]) > 18 else 0.0, 1.0))) for r in rows], dtype=float)
    ttc = np.asarray([max(0.1, 5.0 * (1.0 - min(float(r["x"][19]) if len(r["x"]) > 19 else 0.0, 1.0))) for r in rows], dtype=float)
    hard = np.asarray([r["hardness"] in {"hard", "extreme"} for r in rows], dtype=float)
    failure = np.asarray([r["baseline_failure"] for r in rows], dtype=float)
    return {
        "future_nearest_neighbor_distance": nn,
        "future_ttc_min": ttc,
        "close_pass_event": np.maximum(hard, failure),
        "density_increase_event": hard,
        "crossing_conflict_event": np.asarray(["crossing_paths" in r.get("events", []) for r in rows], dtype=float),
        "stop_go_event": np.asarray(["stop_go" in r.get("events", []) for r in rows], dtype=float),
        "local_congestion_event": np.maximum(hard, failure),
    }


def evaluate(model: Stage7InteractionAuxiliary, rows: List[Dict]) -> Dict:
    if not rows:
        return {}
    targets = make_targets(rows)
    preds = [model.predict(r["x_stage7"]) for r in rows]
    out = {}
    for key, y in targets.items():
        if key in {"future_nearest_neighbor_distance", "future_ttc_min"}:
            p = np.asarray([pred[key] for pred in preds])
            out[f"{key}_MAE"] = round(float(np.mean(np.abs(p - y))), 6)
        else:
            p = np.asarray([pred[f"{key}_prob"] for pred in preds])
            out[f"{key}_AUROC"] = round(float(auroc(y, p)), 6)
            out[f"{key}_AUPRC"] = round(float(auprc(y, p)), 6)
    out["improves_hard_failure_trajectory_performance"] = False
    return out


def report(payload: Dict) -> str:
    rows = [{"metric": k, "value": v} for k, v in payload.get("metrics", {}).items()]
    return "# Stage 7 Interaction Auxiliary Tasks\n\n" + markdown_table(rows) + f"\n\n{payload.get('note', '')}\n"


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"

