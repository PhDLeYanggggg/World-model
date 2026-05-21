from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.evaluation.stage8_goalbench_gold import build_goalbench_gold, write_outputs
from src.models.stage8_goal_predictor_v2 import Stage8GoalPredictorV2, candidate_features, softmax
from src.training.train_goal_intent_predictor import ece


REPORT_DIR = Path("outputs/reports")
CKPT = Path("outputs/checkpoints/stage8/goal_predictor_v2.json")


def load_records() -> List[Dict]:
    p = Path("data/stage8_goalbench_gold/goalbench_gold_records.json")
    if not p.exists():
        write_outputs(build_goalbench_gold())
    return json.loads(p.read_text(encoding="utf-8"))


def train_goal_predictor_v2() -> Dict:
    records = load_records()
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    test = [r for r in records if r["split"] == "test"]
    weights = train_softmax(train)
    model = Stage8GoalPredictorV2({"model": "stage8_goal_predictor_v2", "weights": weights.tolist(), "uses_future_inputs": False})
    model.save(CKPT)
    payload = {
        "checkpoint": str(CKPT),
        "train": evaluate(model, train),
        "val": evaluate(model, val),
        "test": evaluate(model, test),
        "stage7_reference": json.loads((REPORT_DIR / "goal_predictor_metrics_stage7.json").read_text(encoding="utf-8")).get("test", {}) if (REPORT_DIR / "goal_predictor_metrics_stage7.json").exists() else {},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage8_goal_predictor_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORT_DIR / "stage8_goal_predictor_report.md").write_text(report(payload), encoding="utf-8")
    return payload


def train_softmax(rows: List[Dict]) -> np.ndarray:
    w = np.zeros(7, dtype=float)
    usable = [r for r in rows if r.get("true_endpoint_assignment", -1) >= 0 and r.get("candidate_goal_count", 0) > 1]
    for _ in range(700):
        grad = np.zeros_like(w)
        count = 0
        for r in usable:
            feats = candidate_features(r)
            y = int(r["true_endpoint_assignment"])
            if y >= len(feats):
                continue
            p = softmax(feats @ w)
            target = np.zeros(len(p))
            target[y] = 1.0
            grad += feats.T @ (p - target)
            count += 1
        if count:
            w -= 0.05 * (grad / count + 1e-3 * w)
    return w


def evaluate(model: Stage8GoalPredictorV2, rows: List[Dict]) -> Dict:
    usable = [r for r in rows if r.get("true_endpoint_assignment", -1) >= 0 and r.get("candidate_goal_count", 0) > 0]
    if not usable:
        return {"samples": 0}
    labels = [int(r["true_endpoint_assignment"]) for r in usable]
    counts = {l: labels.count(l) for l in set(labels)}
    majority = max(counts, key=counts.get)
    top3maj = {k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:3]}
    top1 = []
    top3 = []
    nll = []
    conf = []
    corr = []
    entropy = []
    for r in usable:
        pred = model.predict(r)
        y = int(r["true_endpoint_assignment"])
        probs = pred["probabilities"]
        top = pred["top_goal_indices"]
        top1.append(float(top and top[0] == y))
        top3.append(float(y in top))
        nll.append(-np.log(max(probs[y] if y < len(probs) else 1e-9, 1e-9)))
        conf.append(max(probs) if probs else 0.0)
        corr.append(top1[-1])
        entropy.append(pred["entropy"])
    maj1 = sum(l == majority for l in labels) / len(labels)
    maj3 = sum(l in top3maj for l in labels) / len(labels)
    hard_rows = [r for r in usable if r.get("agent_count", 1) >= 2]
    return {
        "samples": len(usable),
        "top1_accuracy": round(float(np.mean(top1)), 6),
        "top3_accuracy": round(float(np.mean(top3)), 6),
        "goal_NLL": round(float(np.mean(nll)), 6),
        "goal_ECE": round(float(ece(np.asarray(corr), np.asarray(conf))), 6),
        "goal_entropy": round(float(np.mean(entropy)), 6),
        "majority_top1": round(float(maj1), 6),
        "majority_top3": round(float(maj3), 6),
        "hard_failure_goal_accuracy": round(float(np.mean(top1)) if hard_rows else 0.0, 6),
        "beats_majority": bool(np.mean(top1) > maj1 + 0.02 or (np.mean(top3) > maj3 + 0.02)),
        "top3_saturated": bool(maj3 >= 0.95),
    }


def report(payload: Dict) -> str:
    rows = [{"split": k, **payload[k]} for k in ["train", "val", "test"]]
    return "# Stage 8 Goal Predictor v2\n\n" + markdown_table(rows)


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"

