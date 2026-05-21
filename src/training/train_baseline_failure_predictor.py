from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.models.baseline_failure_predictor import BaselineFailurePredictor
from src.models.stage5b6_gated_residual_model import fit_logistic
from src.training.train_stage5b6_gated_residual import collect_examples, load_json


REPORT_DIR = Path("outputs/reports")
CKPT = Path("outputs/checkpoints/stage6/baseline_failure_predictor.json")


def train_predictor() -> Dict:
    baselines = load_json(REPORT_DIR / "stage5b_baseline_metrics.json", {"datasets": {}})
    train = collect_examples("train", baselines)
    val = collect_examples("val", baselines)
    test = collect_examples("test", baselines)
    x_train = np.stack([r["x"] for r in train]) if train else np.zeros((0, 27))
    mean = x_train.mean(axis=0) if len(x_train) else np.zeros(x_train.shape[1])
    scale = x_train.std(axis=0) if len(x_train) else np.ones(x_train.shape[1])
    xz = (x_train - mean) / np.maximum(scale, 1e-6)
    y = np.asarray([float(r["baseline_failure"]) for r in train], dtype=np.float64)
    weights = np.asarray([2.0 if r["hardness"] == "hard" else 1.0 for r in train], dtype=np.float64)
    coef = fit_logistic(xz, y, weights, steps=800, lr=0.08, l2=1e-3)
    payload = {
        "model": "stage6_baseline_failure_predictor",
        "uses_future_inputs": False,
        "central_velocity_used": False,
        "x_mean": mean.tolist(),
        "x_scale": scale.tolist(),
        "coef": coef.tolist(),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
    }
    model = BaselineFailurePredictor(payload)
    model.save(CKPT)
    metrics = {
        "checkpoint": str(CKPT),
        "train": evaluate_rows(model, train),
        "val": evaluate_rows(model, val),
        "test": evaluate_rows(model, test),
        "by_dataset_test": evaluate_by_dataset(model, test),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "baseline_failure_predictor_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "baseline_failure_predictor_report.md").write_text(report(metrics), encoding="utf-8")
    return metrics


def evaluate_by_dataset(model: BaselineFailurePredictor, rows: List[Dict]) -> Dict:
    out = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        ds = [r for r in rows if r["dataset"] == dataset]
        out[dataset] = evaluate_rows(model, ds)
    return out


def evaluate_rows(model: BaselineFailurePredictor, rows: List[Dict]) -> Dict:
    if not rows:
        return empty_metrics()
    y = np.asarray([float(r["baseline_failure"]) for r in rows], dtype=float)
    p = np.asarray([model.predict_proba(r["x"]) for r in rows], dtype=float)
    pred = p >= 0.5
    easy = [i for i, r in enumerate(rows) if r["hardness"] == "easy"]
    hard = [i for i, r in enumerate(rows) if r["hardness"] == "hard"]
    return {
        "samples": len(rows),
        "positive_rate": round(float(y.mean()), 6),
        "AUROC": round(auroc(y, p), 6),
        "AUPRC": round(auprc(y, p), 6),
        "precision_at_k": round(precision_at_k(y, p, max(1, int(y.sum()))), 6),
        "recall_at_k": round(recall_at_k(y, p, max(1, int(y.sum()))), 6),
        "calibration_ECE": round(ece(y, p), 6),
        "Brier": round(float(np.mean((p - y) ** 2)), 6),
        "easy_false_alarm_rate": round(float(np.mean(pred[easy])) if easy else 0.0, 6),
        "hard_recall": round(float(np.mean(pred[hard] == y[hard])) if hard else 0.0, 6),
        "failure_type_F1": "not_available_no_semantic_ground_truth",
    }


def empty_metrics() -> Dict:
    return {"samples": 0, "positive_rate": 0.0, "AUROC": 0.0, "AUPRC": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0, "calibration_ECE": 1.0, "Brier": 1.0, "easy_false_alarm_rate": 1.0, "hard_recall": 0.0, "failure_type_F1": "n/a"}


def auroc(y, p):
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = sum(float(a > b) + 0.5 * float(a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def auprc(y, p):
    order = np.argsort(-p)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.maximum(np.arange(1, len(y_sorted) + 1), 1)
    recall = tp / max(float(y.sum()), 1.0)
    return float(np.trapz(precision, recall)) if y.sum() else 0.0


def precision_at_k(y, p, k):
    order = np.argsort(-p)[:k]
    return float(y[order].mean()) if len(order) else 0.0


def recall_at_k(y, p, k):
    order = np.argsort(-p)[:k]
    return float(y[order].sum() / max(y.sum(), 1.0))


def ece(y, p, bins=10):
    total = len(y)
    out = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        hi = lo + 1.0 / bins
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return out


def report(metrics: Dict) -> str:
    rows = [{"split": split, **metrics[split]} for split in ["train", "val", "test"]]
    return "# Stage 6 Baseline Failure Predictor\n\n" + markdown_table(rows) + "\n"


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"

