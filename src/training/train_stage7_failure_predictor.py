from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.models.stage5b6_gated_residual_model import fit_logistic
from src.models.stage7_goal_conditioned_failure_predictor import Stage7GoalConditionedFailurePredictor
from src.training.stage7_common import collect_stage7_examples, load_json
from src.training.train_baseline_failure_predictor import auprc, auroc, ece


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage7")
VARIANTS = {
    "without_goal_scene": "no_goal",
    "with_goal_only": "goal_only",
    "with_scene_only": "scene_only",
    "with_goal_scene": "goal_scene",
    "with_goal_scene_interaction": "goal_scene_interaction",
}


def train_stage7_failure_predictors() -> Dict:
    train_by_mode = {mode: collect_stage7_examples("train", feature_mode=mode) for mode in set(VARIANTS.values())}
    val_by_mode = {mode: collect_stage7_examples("val", feature_mode=mode) for mode in set(VARIANTS.values())}
    test_by_mode = {mode: collect_stage7_examples("test", feature_mode=mode) for mode in set(VARIANTS.values())}
    results = {}
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for name, mode in VARIANTS.items():
        model = train_variant(name, mode, train_by_mode[mode])
        path = CKPT_DIR / f"{name}_failure_predictor.json"
        model.save(path)
        results[name] = {
            "checkpoint": str(path),
            "feature_mode": mode,
            "train": evaluate_rows(model, train_by_mode[mode]),
            "val": evaluate_rows(model, val_by_mode[mode]),
            "test": evaluate_rows(model, test_by_mode[mode]),
        }
    stage6 = load_json(REPORT_DIR / "baseline_failure_predictor_metrics.json", {})
    payload = {"stage": "7", "variants": results, "stage6_reference": stage6.get("test", {})}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage7_failure_predictor_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORT_DIR / "stage7_failure_predictor_comparison.md").write_text(report(payload), encoding="utf-8")
    return payload


def train_variant(name: str, mode: str, rows: List[Dict]) -> Stage7GoalConditionedFailurePredictor:
    heads = {}
    for key in sorted({(r["dataset"], r["horizon"]) for r in rows}):
        subset = [r for r in rows if (r["dataset"], r["horizon"]) == key]
        x_raw = np.stack([r["x_stage7"] for r in subset])
        mean = x_raw.mean(axis=0)
        scale = x_raw.std(axis=0)
        y = np.asarray([float(r["baseline_failure"]) for r in subset], dtype=float)
        weights = np.asarray([3.0 if r["baseline_failure"] else (2.0 if r["hardness"] == "hard" else 1.0) for r in subset], dtype=float)
        coef = fit_logistic((x_raw - mean) / np.maximum(scale, 1e-6), y, weights, steps=600, lr=0.08, l2=1e-3)
        heads[f"{key[0]}::{key[1]}"] = {"dataset": key[0], "horizon": key[1], "x_mean": mean.tolist(), "x_scale": scale.tolist(), "coef": coef.tolist()}
    payload = {"model": "stage7_goal_conditioned_failure_predictor", "variant": name, "feature_mode": mode, "uses_future_inputs": False, "heads": heads}
    return Stage7GoalConditionedFailurePredictor(payload)


def evaluate_rows(model: Stage7GoalConditionedFailurePredictor, rows: List[Dict]) -> Dict:
    if not rows:
        return {"samples": 0, "AUROC": 0.5, "AUPRC": 0.0, "calibration_ECE": 1.0, "Brier": 1.0, "hard_recall": 0.0, "easy_false_alarm_rate": 1.0, "failure_type_F1": "not_available"}
    y = np.asarray([float(r["baseline_failure"]) for r in rows], dtype=float)
    p = np.asarray([model.predict_proba(r["dataset"], r["horizon"], r["x_stage7"]) for r in rows], dtype=float)
    pred = p >= 0.5
    hard = np.asarray([r["hardness"] in {"hard", "extreme"} for r in rows], dtype=bool)
    easy = np.asarray([r["hardness"] == "easy" for r in rows], dtype=bool)
    return {
        "samples": len(rows),
        "positive_rate": round(float(y.mean()), 6),
        "AUROC": round(float(auroc(y, p)), 6),
        "AUPRC": round(float(auprc(y, p)), 6),
        "calibration_ECE": round(float(ece(y, p)), 6),
        "Brier": round(float(np.mean((p - y) ** 2)), 6),
        "hard_recall": round(float(np.mean(pred[hard] == y[hard])) if hard.any() else 0.0, 6),
        "easy_false_alarm_rate": round(float(np.mean(pred[easy])) if easy.any() else 0.0, 6),
        "failure_type_F1": "not_available_no_semantic_ground_truth",
    }


def report(payload: Dict) -> str:
    rows = []
    ref = payload.get("stage6_reference", {})
    if ref:
        rows.append({"variant": "stage6_without_goal_scene_reference", **{k: ref.get(k) for k in ["samples", "AUROC", "AUPRC", "calibration_ECE", "Brier", "hard_recall", "easy_false_alarm_rate"]}})
    for name, result in payload["variants"].items():
        rows.append({"variant": name, **result["test"]})
    return "# Stage 7 Failure Predictor Comparison\n\n" + markdown_table(rows)


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"

