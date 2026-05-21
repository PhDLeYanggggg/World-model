from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.models.stage5b6_gated_residual_model import fit_logistic, fit_weighted_ridge
from src.models.stage7_goal_conditioned_world_model import Stage7GoalConditionedWorldModel
from src.training.stage7_common import collect_stage7_examples
from src.training.train_stage7_failure_predictor import train_stage7_failure_predictors


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage7")
VARIANTS = {
    "goal_only_residual": {"feature_mode": "goal_only", "alpha_mode": "goal_conditioned_failure", "hard_weight": 1.5, "failure_weight": 2.0},
    "scene_only_residual": {"feature_mode": "scene_only", "alpha_mode": "goal_conditioned_failure", "hard_weight": 1.5, "failure_weight": 2.0},
    "interaction_scalar_residual": {"feature_mode": "no_goal", "alpha_mode": "learned_alpha_only", "hard_weight": 2.0, "failure_weight": 2.5},
    "goal_interaction_residual": {"feature_mode": "goal_interaction", "alpha_mode": "goal_conditioned_failure", "hard_weight": 2.0, "failure_weight": 3.0},
    "goal_scene_interaction_residual": {"feature_mode": "goal_scene_interaction", "alpha_mode": "goal_conditioned_failure", "hard_weight": 2.0, "failure_weight": 3.0},
    "topk_goal_mixture_diagnostic": {"feature_mode": "goal_scene_interaction", "alpha_mode": "failure_predictor_only", "hard_weight": 2.5, "failure_weight": 3.5},
}


def ensure_failure_predictor() -> str:
    path = CKPT_DIR / "with_goal_scene_interaction_failure_predictor.json"
    if not path.exists():
        train_stage7_failure_predictors()
    return str(path)


def train_stage7_world_models() -> Dict:
    fp_path = ensure_failure_predictor()
    train_cache = {}
    val_cache = {}
    results = []
    for name, cfg in VARIANTS.items():
        mode = cfg["feature_mode"]
        train_cache.setdefault(mode, collect_stage7_examples("train", mode))
        val_cache.setdefault(mode, collect_stage7_examples("val", mode))
        results.append(train_variant(name, cfg, train_cache[mode], val_cache[mode], fp_path))
    payload = {"stage": "7", "variants": results, "latent_enabled": False, "smc_enabled": False}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage7_goal_conditioned_world_model_training.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def train_variant(name: str, cfg: Dict, train_rows: List[Dict], val_rows: List[Dict], fp_path: str) -> Dict:
    heads = {}
    for key in sorted({(r["dataset"], r["horizon"]) for r in train_rows}):
        rows = [r for r in train_rows if (r["dataset"], r["horizon"]) == key]
        x_raw = np.stack([r["x_stage7"] for r in rows])
        mean = x_raw.mean(axis=0)
        scale = x_raw.std(axis=0)
        x = (x_raw - mean) / np.maximum(scale, 1e-6)
        y = np.stack([r["y"] for r in rows])
        failures = np.asarray([float(r["baseline_failure"]) for r in rows])
        weights = np.asarray([example_weight(r, cfg) for r in rows], dtype=float)
        residual_coef = fit_weighted_ridge(x, y, weights, ridge=2e-2)
        alpha_coef = fit_logistic(x, failures, weights, steps=500, lr=0.08, l2=2e-3)
        head = {
            "dataset": key[0],
            "horizon": key[1],
            "x_mean": mean.tolist(),
            "x_scale": scale.tolist(),
            "residual_coef": residual_coef.tolist(),
            "alpha_coef": alpha_coef.tolist(),
            "alpha_scale": 1.0,
            "alpha_bias": 0.0,
        }
        head["alpha_scale"], head["alpha_bias"] = choose_alpha(head, [r for r in val_rows if (r["dataset"], r["horizon"]) == key])
        heads[f"{key[0]}::{key[1]}"] = head
    payload = {
        "model": "stage7_goal_conditioned_world_model",
        "variant": name,
        "feature_mode": cfg["feature_mode"],
        "alpha_mode": cfg["alpha_mode"],
        "failure_predictor_checkpoint": fp_path,
        "residual_clip": 4.0,
        "allow_baseline_fallback": True,
        "latent_enabled": False,
        "smc_enabled": False,
        "heads": heads,
    }
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{name}.json"
    Stage7GoalConditionedWorldModel(payload).save(path)
    return {"variant": name, "checkpoint": str(path), "heads": len(heads), "feature_mode": cfg["feature_mode"]}


def example_weight(row: Dict, cfg: Dict) -> float:
    weight = 1.0
    if row["hardness"] in {"hard", "extreme"}:
        weight *= float(cfg.get("hard_weight", 1.0))
    if row["baseline_failure"]:
        weight *= float(cfg.get("failure_weight", 1.0))
    # High goal entropy should not dominate the residual head.
    if len(row.get("goal_features", [])) and row["goal_features"][2] > 0.8:
        weight *= 0.8
    return float(weight)


def choose_alpha(head: Dict, val_rows: List[Dict]) -> Tuple[float, float]:
    if not val_rows:
        return 0.0, 0.0
    coef = np.asarray(head["residual_coef"], dtype=float)
    acoef = np.asarray(head["alpha_coef"], dtype=float)
    mean = np.asarray(head["x_mean"], dtype=float)
    scale = np.asarray(head["x_scale"], dtype=float)
    best = (0.0, 0.0, float("inf"))
    for alpha_scale in [0.0, 0.2, 0.5, 0.8, 1.0]:
        for alpha_bias in [0.0, 0.02, 0.05]:
            loss = []
            for r in val_rows:
                xb = np.concatenate([[1.0], (r["x_stage7"] - mean) / np.maximum(scale, 1e-6)])
                residual = np.tanh((xb @ coef) / 4.0) * 4.0
                alpha = np.clip(alpha_scale * (1.0 / (1.0 + np.exp(-np.clip(xb @ acoef, -30, 30)))) + alpha_bias, 0.0, 1.0)
                penalty = 0.04 * alpha if r["hardness"] == "easy" and not r["baseline_failure"] else 0.0
                loss.append(float(np.linalg.norm(alpha * residual - r["y"]) + penalty))
            score = float(np.mean(loss))
            if score < best[2]:
                best = (alpha_scale, alpha_bias, score)
    return float(best[0]), float(best[1])

