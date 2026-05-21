from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.evaluation.stage8_goalbench_gold import available_stage8_datasets, load_multiagent_episodes
from src.models.stage5b6_gated_residual_model import fit_logistic, fit_weighted_ridge, sigmoid
from src.models.stage8_multiagent_interaction_encoder import Stage8MultiAgentInteractionEncoder


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage8")


def train_stage8_interaction_auxiliary() -> Dict:
    train = collect_rows("train")
    test = collect_rows("test")
    variants = {}
    for name, mode in {
        "no_interaction": "none",
        "scalar_interaction": "scalar",
        "graph_interaction": "graph",
        "graph_interaction_scene_goal": "graph",
    }.items():
        model = fit_variant(train, mode)
        metrics = evaluate(model, test, mode)
        variants[name] = {"mode": mode, "test": metrics}
    best_scalar = variants["scalar_interaction"]["test"].get("future_nn_mae", 999)
    best_graph = variants["graph_interaction"]["test"].get("future_nn_mae", 999)
    improves = best_graph < best_scalar * 0.97
    payload = {
        "stage": "8",
        "variants": variants,
        "metrics": {
            "graph_improves_over_scalar_auxiliary": bool(improves),
            "improves_hard_failure_trajectory_performance": False,
            "reason": "Stage 8 interaction auxiliary is diagnostic unless benchmark metrics show trajectory lift.",
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage8_interaction_ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORT_DIR / "stage8_interaction_ablation.md").write_text(report(payload), encoding="utf-8")
    return payload


def collect_rows(split: str) -> List[Dict]:
    enc = Stage8MultiAgentInteractionEncoder()
    rows = []
    for dataset in available_stage8_datasets():
        for ep in load_multiagent_episodes(dataset, split=split):
            states = ep["states"]
            meta = ep["meta"]
            past = int(meta["past_horizon"])
            future = states[past:]
            hist = states[:past]
            if future.shape[0] == 0:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "scalar": enc.encode_scalar(hist),
                    "graph": enc.encode_graph(hist),
                    "future_nn": future_nearest_neighbor(future),
                    "close_pass": float(future_nearest_neighbor(future) < 2.0),
                    "density_increase": float(density(future[-1]) > density(hist[-1]) * 1.2),
                }
            )
    return rows


def future_nearest_neighbor(future: np.ndarray) -> float:
    vals = []
    for frame in future:
        pos = frame[:, 0:2]
        valid = np.linalg.norm(pos, axis=1) > 0
        pos = pos[valid]
        if len(pos) < 2:
            continue
        d = np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=2)
        d[d == 0] = np.inf
        vals.append(float(np.min(d)))
    return min(vals) if vals else 999.0


def density(frame: np.ndarray) -> float:
    pos = frame[:, 0:2]
    valid = np.linalg.norm(pos, axis=1) > 0
    pos = pos[valid]
    if len(pos) < 2:
        return 0.0
    bbox = np.maximum(pos.max(axis=0) - pos.min(axis=0), 1.0)
    return float(len(pos) / max(np.prod(bbox), 1.0))


def features(row: Dict, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(1, dtype=float)
    if mode == "scalar":
        return row["scalar"]
    return row["graph"]


def fit_variant(rows: List[Dict], mode: str) -> Dict:
    if not rows:
        return {}
    x_raw = np.stack([features(r, mode) for r in rows])
    mean = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0)
    x = (x_raw - mean) / np.maximum(scale, 1e-6)
    y_reg = np.asarray([[r["future_nn"]] for r in rows], dtype=float)
    y_cls = np.asarray([r["close_pass"] for r in rows], dtype=float)
    coef_reg = fit_weighted_ridge(x, y_reg, np.ones(len(rows)), ridge=1e-2)
    coef_cls = fit_logistic(x, y_cls, np.ones(len(rows)), steps=300, lr=0.05, l2=1e-3)
    return {"mode": mode, "x_mean": mean.tolist(), "x_scale": scale.tolist(), "coef_reg": coef_reg.tolist(), "coef_cls": coef_cls.tolist()}


def evaluate(model: Dict, rows: List[Dict], mode: str) -> Dict:
    if not rows or not model:
        return {"samples": 0}
    mean = np.asarray(model["x_mean"], dtype=float)
    scale = np.asarray(model["x_scale"], dtype=float)
    creg = np.asarray(model["coef_reg"], dtype=float)
    ccls = np.asarray(model["coef_cls"], dtype=float)
    preds = []
    probs = []
    targets = []
    labels = []
    for row in rows:
        x = (features(row, mode) - mean) / np.maximum(scale, 1e-6)
        xb = np.concatenate([[1.0], x])
        preds.append(float((xb @ creg)[0]))
        probs.append(float(sigmoid(xb @ ccls)))
        targets.append(float(row["future_nn"]))
        labels.append(float(row["close_pass"]))
    mae = float(np.mean(np.abs(np.asarray(preds) - np.asarray(targets))))
    y = np.asarray(labels)
    p = np.asarray(probs)
    pred = p >= 0.5
    f1 = float((2 * ((pred == 1) & (y == 1)).sum()) / max((pred == 1).sum() + (y == 1).sum(), 1))
    return {"samples": len(rows), "future_nn_mae": round(mae, 6), "close_pass_F1": round(f1, 6)}


def report(payload: Dict) -> str:
    rows = [{"variant": k, **v.get("test", {})} for k, v in payload.get("variants", {}).items()]
    return "# Stage 8 Interaction v2 Ablation\n\n" + markdown_table(rows) + "\n\nInteraction trajectory lift is determined in `metrics_stage8.json`, not from auxiliary metrics alone.\n"


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"
