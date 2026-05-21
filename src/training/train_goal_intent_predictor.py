from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.evaluation.goalbench_builder import build_goalbench, write_outputs
from src.models.goal_intent_predictor import GoalIntentPredictor, train_softmax


REPORT_DIR = Path("outputs/reports")
CKPT = Path("outputs/checkpoints/stage7/goal_intent_predictor.json")


def load_records() -> List[Dict]:
    path = Path("data/goalbench/goalbench_records.json")
    if not path.exists():
        write_outputs(build_goalbench())
    return json.loads(path.read_text(encoding="utf-8"))


def train_goal_predictor() -> Dict:
    records = load_records()
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    test = [r for r in records if r["split"] == "test"]
    weights = train_softmax(train)
    model = GoalIntentPredictor(
        {
            "model": "stage7_goal_intent_predictor",
            "uses_future_inputs": False,
            "future_endpoint_used_only_as_training_label": True,
            "weights": weights.tolist(),
            "train_records": len(train),
            "val_records": len(val),
            "test_records": len(test),
        }
    )
    model.save(CKPT)
    metrics = {
        "checkpoint": str(CKPT),
        "train": evaluate_goal_records(model, train),
        "val": evaluate_goal_records(model, val),
        "test": evaluate_goal_records(model, test),
        "by_dataset_test": {ds: evaluate_goal_records(model, [r for r in test if r["dataset"] == ds]) for ds in sorted({r["dataset"] for r in test})},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "goal_predictor_metrics_stage7.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "goal_predictor_report_stage7.md").write_text(report(metrics), encoding="utf-8")
    return metrics


def evaluate_goal_records(model: GoalIntentPredictor, rows: List[Dict]) -> Dict:
    usable = [r for r in rows if r.get("true_endpoint_cluster_label", -1) >= 0 and r.get("candidate_goal_count", 0) > 0]
    if not usable:
        return {"samples": 0, "top1_goal_accuracy": 0.0, "top3_goal_accuracy": 0.0, "goal_NLL": 0.0, "goal_ECE": 1.0, "goal_entropy": 0.0, "majority_top1": 0.0, "majority_top3": 0.0}
    labels = [int(r["true_endpoint_cluster_label"]) for r in usable]
    counts = {label: labels.count(label) for label in set(labels)}
    majority = max(counts, key=counts.get)
    top3_majority = {k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:3]}
    top1 = []
    top3 = []
    nll = []
    ent = []
    conf = []
    correct = []
    for r in usable:
        pred = model.predict(r)
        label = int(r["true_endpoint_cluster_label"])
        probs = pred["probabilities"]
        top = pred["top_goal_indices"]
        top1.append(float(top and top[0] == label))
        top3.append(float(label in top))
        nll.append(-np.log(max(probs[label] if label < len(probs) else 1e-9, 1e-9)))
        ent.append(pred["entropy"])
        conf.append(pred["confidence"])
        correct.append(top1[-1])
    return {
        "samples": len(usable),
        "top1_goal_accuracy": round(float(np.mean(top1)), 6),
        "top3_goal_accuracy": round(float(np.mean(top3)), 6),
        "goal_NLL": round(float(np.mean(nll)), 6),
        "goal_ECE": round(ece(np.asarray(correct), np.asarray(conf)), 6),
        "goal_entropy": round(float(np.mean(ent)), 6),
        "majority_top1": round(float(sum(label == majority for label in labels) / len(labels)), 6),
        "majority_top3": round(float(sum(label in top3_majority for label in labels) / len(labels)), 6),
        "beats_majority_top3": bool(np.mean(top3) > (sum(label in top3_majority for label in labels) / len(labels) + 0.02)),
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return 1.0
    out = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        hi = lo + 1.0 / bins
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return out


def report(metrics: Dict) -> str:
    rows = [{"split": split, **metrics[split]} for split in ["train", "val", "test"]]
    rows += [{"split": f"test:{ds}", **payload} for ds, payload in metrics.get("by_dataset_test", {}).items()]
    return "# Stage 7 Goal / Intent Predictor\n\n" + markdown_table(rows) + "\n"


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"

