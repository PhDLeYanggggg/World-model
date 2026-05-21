from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.models.stage5b6_gated_residual_model import fit_logistic
from src.models.stage8_failure_predictor_v2 import Stage8FailurePredictorV2, select_features
from src.training.stage8_common import collect_stage8_examples
from src.training.train_baseline_failure_predictor import auroc, auprc
from src.training.train_goal_intent_predictor import ece


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage8")
VARIANTS = {
    "stage8_without_scene_goal": "no_scene_goal",
    "stage8_with_scene_only": "scene_only",
    "stage8_with_goal_only": "goal_only",
    "stage8_with_scene_goal": "scene_goal",
    "stage8_with_scene_goal_multiagent": "scene_goal_multiagent",
}


def train_stage8_failure_predictors() -> Dict:
    train_rows = collect_stage8_examples("train")
    val_rows = collect_stage8_examples("val")
    test_rows = collect_stage8_examples("test")
    results = {}
    for name, mode in VARIANTS.items():
        model = fit_variant(name, mode, train_rows)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        path = CKPT_DIR / f"{name}.json"
        model.save(path)
        results[name] = {
            "checkpoint": str(path),
            "feature_mode": mode,
            "train": evaluate(model, train_rows),
            "val": evaluate(model, val_rows),
            "test": evaluate(model, test_rows),
        }
    stage7_ref = {}
    ref_path = REPORT_DIR / "stage7_failure_predictor_comparison.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        best = max((v.get("test", {}).get("AUROC", 0.0) for v in ref.get("variants", {}).values()), default=0.0)
        stage7_ref = {"best_stage7_test_AUROC": best, "source": str(ref_path)}
    payload = {"stage": "8", "variants": results, "stage7_reference": stage7_ref, "uses_future_inputs": False}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage8_failure_predictor_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORT_DIR / "stage8_failure_predictor_comparison.md").write_text(report(payload), encoding="utf-8")
    return payload


def fit_variant(name: str, mode: str, rows: List[Dict]) -> Stage8FailurePredictorV2:
    heads = {}
    for key in sorted({(r["dataset"], r["horizon"]) for r in rows}):
        sub = [r for r in rows if (r["dataset"], r["horizon"]) == key]
        if not sub:
            continue
        x_raw = np.stack([select_features(r["x"], mode) for r in sub])
        mean = x_raw.mean(axis=0)
        scale = x_raw.std(axis=0)
        x = (x_raw - mean) / np.maximum(scale, 1e-6)
        y = np.asarray([float(r["baseline_failure"]) for r in sub], dtype=float)
        weights = np.asarray([2.5 if yv > 0.5 else 1.0 for yv in y], dtype=float)
        coef = fit_logistic(x, y, weights, steps=500, lr=0.08, l2=2e-3)
        heads[f"{key[0]}::{key[1]}"] = {"dataset": key[0], "horizon": key[1], "feature_mode": mode, "x_mean": mean.tolist(), "x_scale": scale.tolist(), "coef": coef.tolist()}
    return Stage8FailurePredictorV2(
        {
            "model": "stage8_failure_predictor_v2",
            "variant": name,
            "feature_mode": mode,
            "heads": heads,
            "uses_future_endpoint_as_input": False,
            "uses_central_velocity": False,
        }
    )


def evaluate(model: Stage8FailurePredictorV2, rows: List[Dict]) -> Dict:
    if not rows:
        return {"samples": 0}
    y = np.asarray([float(r["baseline_failure"]) for r in rows], dtype=float)
    p = np.asarray([model.predict_proba(r["dataset"], r["horizon"], r["x"]) for r in rows], dtype=float)
    hard = np.asarray([1.0 if r["hardness"] == "hard" else 0.0 for r in rows], dtype=float)
    hard_mask = hard > 0.5
    easy_mask = ~hard_mask
    pred = p >= 0.5
    return {
        "samples": int(len(rows)),
        "failure_rate": round(float(y.mean()) if len(y) else 0.0, 6),
        "AUROC": round(float(auroc(y, p)), 6),
        "AUPRC": round(float(auprc(y, p)), 6),
        "Brier": round(float(np.mean((p - y) ** 2)), 6),
        "ECE": round(float(ece(y, p)), 6),
        "hard_recall": round(float(((pred & (y > 0.5) & hard_mask).sum() / max(((y > 0.5) & hard_mask).sum(), 1))), 6),
        "easy_false_alarm_rate": round(float(((pred & (y < 0.5) & easy_mask).sum() / max(((y < 0.5) & easy_mask).sum(), 1))), 6),
    }


def report(payload: Dict) -> str:
    rows = []
    for name, item in payload.get("variants", {}).items():
        rows.append({"variant": name, "feature_mode": item.get("feature_mode"), **item.get("test", {})})
    return "# Stage 8 Failure Predictor v2\n\n" + markdown_table(rows)


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"
