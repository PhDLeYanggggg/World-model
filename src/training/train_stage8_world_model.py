from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.models.stage5b6_gated_residual_model import fit_logistic, fit_weighted_ridge, sigmoid
from src.models.stage8_failure_predictor_v2 import select_features
from src.models.stage8_goal_conditioned_world_model_v2 import Stage8GoalConditionedWorldModelV2
from src.training.stage8_common import collect_stage8_examples
from src.training.train_stage8_failure_predictor import train_stage8_failure_predictors


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage8")
VARIANTS = {
    "stage7_best_model_reference": {"feature_mode": "scene_goal_multiagent", "reference_only": True},
    "goal_only_v2": {"feature_mode": "goal_only", "hard_weight": 1.5, "failure_weight": 2.0},
    "scene_only_v2": {"feature_mode": "scene_only", "hard_weight": 1.5, "failure_weight": 2.0},
    "multiagent_only_v2": {"feature_mode": "no_scene_goal", "hard_weight": 1.8, "failure_weight": 2.4},
    "scene_goal_v2": {"feature_mode": "scene_goal", "hard_weight": 2.0, "failure_weight": 2.8},
    "scene_goal_multiagent_v2": {"feature_mode": "scene_goal_multiagent", "hard_weight": 2.2, "failure_weight": 3.0},
    "topk_goal_diagnostic_v2": {"feature_mode": "scene_goal_multiagent", "hard_weight": 2.5, "failure_weight": 3.5, "diagnostic_only": True},
}


def ensure_failure_predictor() -> str:
    path = CKPT_DIR / "stage8_with_scene_goal_multiagent.json"
    if not path.exists():
        train_stage8_failure_predictors()
    return str(path)


def train_stage8_world_models() -> Dict:
    fp_path = ensure_failure_predictor()
    train_rows = collect_stage8_examples("train")
    val_rows = collect_stage8_examples("val")
    results = []
    for name, cfg in VARIANTS.items():
        if cfg.get("reference_only"):
            results.append({"variant": name, "checkpoint": "outputs/checkpoints/stage7/goal_scene_interaction_residual.json", "reference_only": True})
            continue
        results.append(train_variant(name, cfg, train_rows, val_rows, fp_path))
    payload = {"stage": "8", "variants": results, "latent_enabled": False, "smc_enabled": False}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage8_goal_conditioned_world_model_training.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def train_variant(name: str, cfg: Dict, train_rows: List[Dict], val_rows: List[Dict], fp_path: str) -> Dict:
    mode = cfg["feature_mode"]
    heads = {}
    for key in sorted({(r["dataset"], r["horizon"]) for r in train_rows}):
        rows = [r for r in train_rows if (r["dataset"], r["horizon"]) == key]
        if not rows:
            continue
        x_raw = np.stack([select_features(r["x"], mode) for r in rows])
        mean = x_raw.mean(axis=0)
        scale = x_raw.std(axis=0)
        x = (x_raw - mean) / np.maximum(scale, 1e-6)
        y = np.stack([r["y"] for r in rows])
        failures = np.asarray([float(r["baseline_failure"]) for r in rows], dtype=float)
        weights = np.asarray([example_weight(r, cfg) for r in rows], dtype=float)
        residual_coef = fit_weighted_ridge(x, y, weights, ridge=3e-2)
        alpha_coef = fit_logistic(x, failures, weights, steps=500, lr=0.08, l2=2e-3)
        head = {
            "dataset": key[0],
            "horizon": key[1],
            "feature_mode": mode,
            "x_mean": mean.tolist(),
            "x_scale": scale.tolist(),
            "residual_coef": residual_coef.tolist(),
            "alpha_coef": alpha_coef.tolist(),
            "alpha_scale": 1.0,
            "alpha_bias": 0.0,
        }
        head["alpha_scale"], head["alpha_bias"] = choose_alpha(head, [r for r in val_rows if (r["dataset"], r["horizon"]) == key], mode)
        heads[f"{key[0]}::{key[1]}"] = head
    payload = {
        "model": "stage8_goal_conditioned_world_model_v2",
        "variant": name,
        "feature_mode": mode,
        "failure_predictor_checkpoint": fp_path,
        "residual_clip": 3.0,
        "allow_baseline_fallback": True,
        "diagnostic_only": bool(cfg.get("diagnostic_only", False)),
        "latent_enabled": False,
        "smc_enabled": False,
        "heads": heads,
    }
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{name}.json"
    Stage8GoalConditionedWorldModelV2(payload).save(path)
    return {"variant": name, "checkpoint": str(path), "heads": len(heads), "feature_mode": mode, "diagnostic_only": bool(cfg.get("diagnostic_only", False))}


def example_weight(row: Dict, cfg: Dict) -> float:
    weight = 1.0
    if row["hardness"] == "hard":
        weight *= float(cfg.get("hard_weight", 1.0))
    if row["baseline_failure"]:
        weight *= float(cfg.get("failure_weight", 1.0))
    # Do not overfit unconfirmed inferred goals.
    if len(row["x"]) >= 23 and row["x"][22] > 0.5:
        weight *= 0.85
    return float(weight)


def choose_alpha(head: Dict, val_rows: List[Dict], mode: str) -> Tuple[float, float]:
    if not val_rows:
        return 0.0, 0.0
    coef = np.asarray(head["residual_coef"], dtype=float)
    acoef = np.asarray(head["alpha_coef"], dtype=float)
    mean = np.asarray(head["x_mean"], dtype=float)
    scale = np.asarray(head["x_scale"], dtype=float)
    best = (0.0, 0.0, float("inf"))
    for alpha_scale in [0.0, 0.15, 0.3, 0.5, 0.8, 1.0]:
        for alpha_bias in [0.0, 0.02, 0.05]:
            losses = []
            for row in val_rows:
                x_use = select_features(row["x"], mode)
                xb = np.concatenate([[1.0], (x_use - mean) / np.maximum(scale, 1e-6)])
                residual = np.tanh((xb @ coef) / 3.0) * 3.0
                alpha = np.clip(alpha_scale * sigmoid(xb @ acoef) + alpha_bias, 0.0, 1.0)
                easy_penalty = 0.05 * alpha if not row["baseline_failure"] and row["hardness"] == "easy" else 0.0
                losses.append(float(np.linalg.norm(alpha * residual - row["y"]) + easy_penalty))
            score = float(np.mean(losses))
            if score < best[2]:
                best = (alpha_scale, alpha_bias, score)
    return float(best[0]), float(best[1])
