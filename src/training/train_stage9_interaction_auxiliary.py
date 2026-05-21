from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.evaluation.stage9_data_audit import load_stage9_episodes
from src.models.stage9_interaction_encoder import encode_interaction
from src.models.stage9_residual_decoder import fit_logistic, fit_ridge, sigmoid


REPORT_DIR = Path("outputs/reports")


def collect_aux_rows(split: str) -> List[Dict]:
    rows = []
    for ep in load_stage9_episodes(split=split):
        past = int(ep["meta"]["past_horizon"])
        hist = ep["states"][:past]
        hist_mask = ep["mask"][:past]
        future = ep["states"][past:]
        future_mask = ep["mask"][past:]
        for agent_idx in np.where(hist_mask[-1])[0]:
            rows.append(
                {
                    "dataset": ep["meta"]["dataset_name"],
                    "x": encode_interaction(hist, hist_mask, int(agent_idx)),
                    "future_nn": future_nearest_neighbor(future, future_mask, int(agent_idx)),
                    "close_pass": 1.0 if future_nearest_neighbor(future, future_mask, int(agent_idx)) < 2.0 else 0.0,
                    "density_increase": 1.0 if density(future[-1], future_mask[-1]) > density(hist[-1], hist_mask[-1]) * 1.2 else 0.0,
                    "congestion": 1.0 if density(future[-1], future_mask[-1]) > 0.2 else 0.0,
                }
            )
    return rows


def train_stage9_interaction_auxiliary() -> Dict:
    train = collect_aux_rows("train")
    test = collect_aux_rows("test") or collect_aux_rows("val")
    model = fit_aux(train)
    metrics = evaluate_aux(model, test)
    payload = {
        "stage": "9",
        "model": model,
        "test": metrics,
        "trajectory_lift": "evaluated_by_stage9_benchmark_not_auxiliary",
        "diagnostic_only_if_no_benchmark_lift": True,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage9_interaction_auxiliary_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORT_DIR / "stage9_interaction_auxiliary_report.md").write_text(report(payload), encoding="utf-8")
    return payload


def future_nearest_neighbor(future: np.ndarray, mask: np.ndarray, agent_idx: int) -> float:
    vals = []
    for frame, m in zip(future, mask):
        if agent_idx >= len(m) or not m[agent_idx]:
            continue
        others = m.copy()
        others[agent_idx] = False
        pos = frame[others, 0:2]
        if len(pos) == 0:
            continue
        vals.append(float(np.min(np.linalg.norm(pos - frame[agent_idx, 0:2][None, :], axis=1))))
    return min(vals) if vals else 999.0


def density(frame: np.ndarray, mask: np.ndarray) -> float:
    pos = frame[mask, 0:2]
    if len(pos) < 2:
        return 0.0
    bbox = np.maximum(pos.max(axis=0) - pos.min(axis=0), 1.0)
    return float(len(pos) / max(np.prod(bbox), 1.0))


def fit_aux(rows: List[Dict]) -> Dict:
    if not rows:
        return {}
    x_raw = np.stack([r["x"] for r in rows])
    mean = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0)
    x = (x_raw - mean) / np.maximum(scale, 1e-6)
    y_reg = np.asarray([[r["future_nn"]] for r in rows])
    y_close = np.asarray([r["close_pass"] for r in rows])
    y_density = np.asarray([r["density_increase"] for r in rows])
    weights = np.ones(len(rows))
    return {
        "x_mean": mean.tolist(),
        "x_scale": scale.tolist(),
        "future_nn_coef": fit_ridge(x, y_reg, weights, ridge=2e-2).tolist(),
        "close_pass_coef": fit_logistic(x, y_close, weights, steps=300, lr=0.05).tolist(),
        "density_increase_coef": fit_logistic(x, y_density, weights, steps=300, lr=0.05).tolist(),
    }


def evaluate_aux(model: Dict, rows: List[Dict]) -> Dict:
    if not rows or not model:
        return {"samples": 0}
    mean = np.asarray(model["x_mean"], dtype=float)
    scale = np.asarray(model["x_scale"], dtype=float)
    nn_coef = np.asarray(model["future_nn_coef"], dtype=float)
    close_coef = np.asarray(model["close_pass_coef"], dtype=float)
    den_coef = np.asarray(model["density_increase_coef"], dtype=float)
    nn_pred, nn_true, close_p, close_y, den_p, den_y = [], [], [], [], [], []
    for r in rows:
        x = (r["x"] - mean) / np.maximum(scale, 1e-6)
        xb = np.concatenate([[1.0], x])
        nn_pred.append(float((xb @ nn_coef)[0]))
        nn_true.append(float(r["future_nn"]))
        close_p.append(float(sigmoid(xb @ close_coef)))
        close_y.append(float(r["close_pass"]))
        den_p.append(float(sigmoid(xb @ den_coef)))
        den_y.append(float(r["density_increase"]))
    return {
        "samples": len(rows),
        "future_nearest_neighbor_MAE": round(float(np.mean(np.abs(np.asarray(nn_pred) - np.asarray(nn_true)))), 6),
        "close_pass_F1": round(f1(np.asarray(close_y), np.asarray(close_p) >= 0.5), 6),
        "density_increase_F1": round(f1(np.asarray(den_y), np.asarray(den_p) >= 0.5), 6),
    }


def f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(((y > 0.5) & pred).sum())
    denom = float(pred.sum() + (y > 0.5).sum())
    return 2 * tp / max(denom, 1.0)


def report(payload: Dict) -> str:
    return "# Stage 9 Interaction Auxiliary v3\n\n" + json.dumps(payload["test"], indent=2) + "\n\nAuxiliary success is diagnostic unless Stage 9 benchmark shows trajectory lift.\n"
