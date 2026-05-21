from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.models.baseline_failure_predictor import BaselineFailurePredictor
from src.models.stage5b6_gated_residual_model import fit_logistic, fit_weighted_ridge, trim_features_for_mode
from src.training.train_stage5b6_gated_residual import collect_examples, example_weight, load_json


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage6")
FP_CKPT = CKPT_DIR / "baseline_failure_predictor.json"
VARIANTS = {
    "failure_predictor_only_gate": {"alpha_mode": "failure_predictor_only", "feature_mode": "full", "hard_weight": 2.0, "failure_weight": 3.0},
    "learned_alpha_gate": {"alpha_mode": "learned_alpha", "feature_mode": "full", "hard_weight": 2.0, "failure_weight": 2.0},
    "hybrid_failure_predictor_plus_learned_gate": {"alpha_mode": "hybrid", "feature_mode": "full", "hard_weight": 3.0, "failure_weight": 3.0},
    "no_interaction_ablation": {"alpha_mode": "hybrid", "feature_mode": "no_interaction", "hard_weight": 2.0, "failure_weight": 2.0},
    "scalar_interaction_ablation": {"alpha_mode": "hybrid", "feature_mode": "scalar_interaction", "hard_weight": 2.0, "failure_weight": 2.0},
    "graph_interaction_ablation": {"alpha_mode": "hybrid", "feature_mode": "full", "hard_weight": 2.0, "failure_weight": 2.0},
}


def choose_scale(head: Dict, val_rows: List[Dict], feature_mode: str, alpha_mode: str, fp: BaselineFailurePredictor) -> Tuple[float, float]:
    if not val_rows:
        return 0.0, 0.0
    coef = np.asarray(head["residual_coef"], dtype=np.float64)
    alpha_coef = np.asarray(head["alpha_coef"], dtype=np.float64)
    mean = np.asarray(head["x_mean"], dtype=np.float64)
    scale = np.asarray(head["x_scale"], dtype=np.float64)
    best = (0.0, 0.0, float("inf"))
    for alpha_scale in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for alpha_bias in [0.0, 0.03, 0.08]:
            losses = []
            for row in val_rows:
                x_use = trim_features_for_mode(row["x"], feature_mode)
                xb = np.concatenate([[1.0], (x_use - mean) / np.maximum(scale, 1e-6)])
                residual = np.tanh((xb @ coef) / 4.0) * 4.0
                learned = 1.0 / (1.0 + np.exp(-np.clip(xb @ alpha_coef, -30, 30)))
                fp_prob = fp.predict_proba(row["x"])
                if alpha_mode == "failure_predictor_only":
                    alpha = fp_prob
                elif alpha_mode == "learned_alpha":
                    alpha = learned
                else:
                    alpha = 0.7 * fp_prob + 0.3 * learned
                alpha = np.clip(alpha_scale * alpha + alpha_bias, 0.0, 1.0)
                easy_penalty = 0.04 * alpha if row["hardness"] == "easy" and not row["baseline_failure"] else 0.0
                losses.append(float(np.linalg.norm(alpha * residual - row["y"]) + easy_penalty))
            score = float(np.mean(losses))
            if score < best[2]:
                best = (alpha_scale, alpha_bias, score)
    return float(best[0]), float(best[1])


def train_variant(name: str, cfg: Dict, train_rows: List[Dict], val_rows: List[Dict], fp: BaselineFailurePredictor) -> Dict:
    feature_mode = cfg["feature_mode"]
    heads = {}
    for key in sorted({(r["dataset"], r["horizon"]) for r in train_rows}):
        rows = [r for r in train_rows if (r["dataset"], r["horizon"]) == key]
        x_raw = np.stack([trim_features_for_mode(r["x"], feature_mode) for r in rows])
        mean = x_raw.mean(axis=0)
        scale = x_raw.std(axis=0)
        x = (x_raw - mean) / np.maximum(scale, 1e-6)
        y = np.stack([r["y"] for r in rows])
        weights = np.asarray([example_weight(r, cfg) for r in rows])
        failures = np.asarray([float(r["baseline_failure"]) for r in rows])
        residual_coef = fit_weighted_ridge(x, y, weights, ridge=3e-2)
        alpha_coef = fit_logistic(x, failures, weights, steps=500, lr=0.08, l2=2e-3)
        head = {
            "dataset": key[0],
            "horizon": key[1],
            "feature_mode": feature_mode,
            "x_mean": mean.tolist(),
            "x_scale": scale.tolist(),
            "residual_coef": residual_coef.tolist(),
            "alpha_coef": alpha_coef.tolist(),
            "alpha_scale": 1.0,
            "alpha_bias": 0.0,
        }
        matching_val = [r for r in val_rows if (r["dataset"], r["horizon"]) == key]
        head["alpha_scale"], head["alpha_bias"] = choose_scale(head, matching_val, feature_mode, cfg["alpha_mode"], fp)
        heads[f"{key[0]}::{key[1]}"] = head
    payload = {
        "model": "stage6_failure_aware_gated_residual",
        "variant": name,
        "alpha_mode": cfg["alpha_mode"],
        "feature_mode": feature_mode,
        "latent_enabled": False,
        "smc_enabled": False,
        "residual_clip": 4.0,
        "failure_predictor_checkpoint": str(FP_CKPT),
        "heads": heads,
    }
    path = CKPT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"variant": name, "checkpoint": str(path), "heads": len(heads), "nonzero_alpha_heads": sum(1 for h in heads.values() if h["alpha_scale"] > 0 or h["alpha_bias"] > 0)}


def train_failure_aware_models() -> Dict:
    baselines = load_json(REPORT_DIR / "stage5b_baseline_metrics.json", {"datasets": {}})
    train = collect_examples("train", baselines)
    val = collect_examples("val", baselines)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if not FP_CKPT.exists():
        from src.training.train_baseline_failure_predictor import train_predictor

        train_predictor()
    fp = BaselineFailurePredictor.load(FP_CKPT)
    variants = [train_variant(name, cfg, train, val, fp) for name, cfg in VARIANTS.items()]
    payload = {"stage": "6", "variants": variants, "train_examples": len(train), "val_examples": len(val)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage6_failure_aware_training.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

