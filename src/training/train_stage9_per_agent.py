from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.evaluation.stage9_data_audit import load_stage9_episodes
from src.evaluation.stage9_per_agent_baselines import rollout_baseline, run_stage9_baselines, write_stage9_baselines
from src.models.stage9_agent_encoder import encode_agent_history
from src.models.stage9_goal_encoder import encode_goal
from src.models.stage9_interaction_encoder import encode_interaction
from src.models.stage9_per_agent_world_model import Stage9PerAgentWorldModel
from src.models.stage9_residual_decoder import fit_logistic, fit_ridge, sigmoid
from src.models.stage9_scene_encoder import encode_scene, load_scene_pack


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage9")
VARIANTS = {
    "per_agent_no_scene": "no_scene",
    "per_agent_scene_only": "scene_only",
    "per_agent_goal_only": "goal_only",
    "per_agent_interaction_only": "interaction_only",
    "per_agent_scene_goal": "scene_goal",
    "per_agent_full_scene_goal_interaction": "full",
}


def ensure_baselines() -> Dict:
    path = REPORT_DIR / "stage9_per_agent_baseline_metrics.json"
    if not path.exists():
        payload = run_stage9_baselines()
        write_stage9_baselines(payload)
    return json.loads(path.read_text(encoding="utf-8"))


def collect_examples(split: str, baselines: Dict) -> List[Dict]:
    rows = []
    for dataset, drow in baselines.get("datasets", {}).items():
        baseline_name = drow["strongest_causal_baseline"]
        for ep in load_stage9_episodes(dataset, split=split):
            future = int(ep["meta"].get("future_horizon", 0))
            horizons = [h for h in [1, 5, 10, 25, 50, 100] if h <= future]
            for horizon in horizons:
                base = rollout_baseline(ep, horizon, baseline_name)
                past = int(ep["meta"]["past_horizon"])
                true = ep["states"][past : past + horizon]
                valid = ep["mask"][past + horizon - 1]
                for agent_idx in np.where(valid)[0]:
                    if not ep["mask"][past - 1, agent_idx]:
                        continue
                    x_full = stage9_feature_vector(ep, base, int(agent_idx), horizon)
                    y = true[horizon - 1, agent_idx, 0:2] - base[horizon - 1, agent_idx, 0:2]
                    b_fde = float(np.linalg.norm(y))
                    rows.append(
                        {
                            "dataset": dataset,
                            "scene_id": ep["meta"]["scene_id"],
                            "episode_id": int(ep["meta"]["episode_id"]),
                            "agent_idx": int(agent_idx),
                            "horizon": horizon,
                            "x_full": x_full,
                            "y": y.astype(float),
                            "baseline_fde": b_fde,
                            "hard": bool(ep["meta"].get("hard_interaction")),
                            "baseline_failure": bool(ep["meta"].get("baseline_failure_proxy") or b_fde > failure_threshold(dataset, horizon)),
                            "annotation_quality": ep["meta"].get("annotation_quality", "not_available"),
                            "agent_count": int(ep["meta"].get("agent_count", 0)),
                        }
                    )
    return rows


def stage9_feature_vector(ep: Dict, baseline: np.ndarray, agent_idx: int, horizon: int) -> np.ndarray:
    past = int(ep["meta"]["past_horizon"])
    hist = ep["states"][:past]
    hist_mask = ep["mask"][:past]
    dataset = ep["meta"]["dataset_name"]
    scene_id = ep["meta"]["scene_id"]
    pack = load_scene_pack(dataset, scene_id)
    last = hist[-1, agent_idx]
    b_end = baseline[horizon - 1, agent_idx]
    base_feat = np.asarray(
        [
            float(b_end[0] - last[0]) / 50.0,
            float(b_end[1] - last[1]) / 50.0,
            float(np.linalg.norm(baseline[:, agent_idx, 2:4], axis=1).mean()) / 10.0,
            float(horizon / max(ep["meta"].get("future_horizon", horizon), 1)),
            1.0 if ep["meta"].get("coordinate_unit") == "meter" else 0.0,
        ],
        dtype=float,
    )
    return np.concatenate(
        [
            encode_agent_history(hist, hist_mask, agent_idx, horizon, max_horizon=int(ep["meta"].get("future_horizon", horizon))),
            base_feat,
            encode_scene(pack, last[0:2]),
            encode_goal(pack, last[0:2], last[2:4]),
            encode_interaction(hist, hist_mask, agent_idx),
        ]
    )


def select_features(x: np.ndarray, mode: str) -> np.ndarray:
    # full layout: agent12 | baseline5 | scene6 | goal8 | interaction10
    agent_base = x[:17]
    scene = x[17:23]
    goal = x[23:31]
    inter = x[31:41]
    if mode == "no_scene":
        return agent_base
    if mode == "scene_only":
        return np.concatenate([agent_base, scene])
    if mode == "goal_only":
        return np.concatenate([agent_base, goal])
    if mode == "interaction_only":
        return np.concatenate([agent_base, inter])
    if mode == "scene_goal":
        return np.concatenate([agent_base, scene, goal])
    return np.concatenate([agent_base, scene, goal, inter])


def failure_threshold(dataset: str, horizon: int) -> float:
    if dataset in {"trajnet", "eth_ucy"}:
        return 1.0 if horizon <= 10 else 2.0
    return 3.0


def train_stage9_models() -> Dict:
    baselines = ensure_baselines()
    train_rows = collect_examples("train", baselines)
    val_rows = collect_examples("val", baselines)
    results = []
    for name, mode in VARIANTS.items():
        results.append(train_variant(name, mode, train_rows, val_rows))
    payload = {"stage": "9", "variants": results, "latent_enabled": False, "smc_enabled": False, "predicts_all_agents": True}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage9_per_agent_training.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def train_variant(name: str, mode: str, train_rows: List[Dict], val_rows: List[Dict]) -> Dict:
    heads = {}
    for key in sorted({(r["dataset"], r["horizon"]) for r in train_rows}):
        rows = [r for r in train_rows if (r["dataset"], r["horizon"]) == key]
        x_raw = np.stack([select_features(r["x_full"], mode) for r in rows])
        mean = x_raw.mean(axis=0)
        scale = x_raw.std(axis=0)
        x = (x_raw - mean) / np.maximum(scale, 1e-6)
        y = np.stack([r["y"] for r in rows])
        failures = np.asarray([float(r["baseline_failure"] or r["hard"]) for r in rows])
        weights = np.asarray([example_weight(r) for r in rows], dtype=float)
        rcoef = fit_ridge(x, y, weights, ridge=2e-2)
        acoef = fit_logistic(x, failures, weights, steps=500, lr=0.08, l2=2e-3)
        head = {
            "dataset": key[0],
            "horizon": key[1],
            "x_mean": mean.tolist(),
            "x_scale": scale.tolist(),
            "residual_coef": rcoef.tolist(),
            "alpha_coef": acoef.tolist(),
            "alpha_scale": 1.0,
            "alpha_bias": 0.0,
        }
        head["alpha_scale"], head["alpha_bias"] = choose_alpha(head, [r for r in val_rows if (r["dataset"], r["horizon"]) == key], mode)
        heads[f"{key[0]}::{key[1]}"] = head
    payload = {
        "model": "stage9_per_agent_world_model",
        "variant": name,
        "feature_mode": mode,
        "residual_clip": 2.0,
        "heads": heads,
        "predicts_all_agents": True,
        "latent_enabled": False,
        "smc_enabled": False,
    }
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{name}.json"
    Stage9PerAgentWorldModel(payload).save(path)
    return {"variant": name, "checkpoint": str(path), "heads": len(heads), "feature_mode": mode}


def example_weight(row: Dict) -> float:
    weight = 1.0
    if row["hard"]:
        weight *= 1.8
    if row["baseline_failure"]:
        weight *= 2.5
    if row["annotation_quality"] == "silver":
        weight *= 1.1
    return float(weight)


def choose_alpha(head: Dict, rows: List[Dict], mode: str) -> Tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    mean = np.asarray(head["x_mean"], dtype=float)
    scale = np.asarray(head["x_scale"], dtype=float)
    rcoef = np.asarray(head["residual_coef"], dtype=float)
    acoef = np.asarray(head["alpha_coef"], dtype=float)
    best = (0.0, 0.0, float("inf"))
    for alpha_scale in [0.0, 0.15, 0.3, 0.5, 0.8, 1.0]:
        for alpha_bias in [0.0, 0.02, 0.05]:
            losses = []
            for row in rows:
                x = select_features(row["x_full"], mode)
                xb = np.concatenate([[1.0], (x - mean) / np.maximum(scale, 1e-6)])
                raw = xb @ rcoef
                residual = np.tanh(raw / 2.0) * 2.0
                alpha = np.clip(alpha_scale * sigmoid(xb @ acoef) + alpha_bias, 0.0, 1.0)
                penalty = 0.03 * alpha if not row["hard"] and not row["baseline_failure"] else 0.0
                losses.append(float(np.linalg.norm(alpha * residual - row["y"]) + penalty))
            score = float(np.mean(losses))
            if score < best[2]:
                best = (alpha_scale, alpha_bias, score)
    return float(best[0]), float(best[1])
