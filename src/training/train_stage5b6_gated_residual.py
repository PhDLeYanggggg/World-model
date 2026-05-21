from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.evaluation.baseline_benchmark_stage5b import load_dataset_episodes, rollout
from src.evaluation.baseline_failure_oracle import threshold_for
from src.models.encoders.stage5b6_interaction_encoder import Stage5B6InteractionEncoder
from src.models.stage5b6_gated_residual_model import fit_logistic, fit_weighted_ridge, trim_features_for_mode


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage5b6")
VARIANTS = {
    "gated_residual_all_data": {"hard_weight": 1.0, "failure_weight": 1.0, "feature_mode": "full"},
    "gated_residual_hard_weighted": {"hard_weight": 3.0, "failure_weight": 1.5, "feature_mode": "full"},
    "gated_residual_failure_classifier_aux": {"hard_weight": 2.0, "failure_weight": 3.0, "feature_mode": "full"},
    "no_interaction_ablation": {"hard_weight": 2.0, "failure_weight": 2.0, "feature_mode": "no_interaction"},
    "nearest_neighbor_scalar_ablation": {"hard_weight": 2.0, "failure_weight": 2.0, "feature_mode": "scalar_interaction"},
    "graph_attention_interaction_ablation": {"hard_weight": 2.0, "failure_weight": 2.0, "feature_mode": "full"},
    "graph_attention_temporal_history_ablation": {"hard_weight": 2.0, "failure_weight": 2.0, "feature_mode": "full"},
}


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def hard_lookup() -> Dict[Tuple[str, int], Dict]:
    rows = load_json(REPORT_DIR / "stage5b5_hard_subset_summary.json", [])
    out = {}
    for dataset in rows:
        for ep in dataset.get("episodes", []):
            out[(dataset["dataset_name"], int(ep["episode_id"]))] = ep
    return out


def stage5b6_feature_vector(dataset: str, hist: np.ndarray, meta: Dict, baseline_name: str, horizon: int, future_len: int, interaction: np.ndarray) -> np.ndarray:
    last = hist[-1, 0]
    prev = hist[-2, 0] if hist.shape[0] >= 2 else last
    heading_delta = float(np.angle(np.exp(1j * (last[6] - prev[6]))))
    speed_change = float(last[7] - prev[7])
    headings = np.unwrap(np.arctan2(hist[:, 0, 3], hist[:, 0, 2]))
    speeds = np.linalg.norm(hist[:, 0, 2:4], axis=1)
    past_curvature = float(np.sum(np.abs(np.diff(headings)))) if len(headings) > 1 else 0.0
    past_speed_total = float(np.sum(np.abs(np.diff(speeds)))) if len(speeds) > 1 else 0.0
    past_self_error = past_baseline_self_error(hist, float(meta.get("dt_s", 1.0)), baseline_name)
    base_endpoint_disp = float(np.linalg.norm(rollout(hist, horizon, float(meta.get("dt_s", 1.0)), baseline_name)[-1, 0, 0:2] - last[0:2]))
    is_traffic = 1.0 if dataset.startswith("tgsim") else 0.0
    is_ped = 1.0 if dataset in {"trajnet", "eth_ucy"} else 0.0
    coord_metric = 1.0 if meta.get("coordinate_unit") == "meter" else 0.0
    base = np.asarray(
        [
            last[2],
            last[3],
            last[4],
            last[5],
            last[7],
            np.sin(last[6]),
            np.cos(last[6]),
            heading_delta,
            speed_change,
            np.linalg.norm(last[4:6]),
            min(past_curvature, 10.0) / 10.0,
            min(past_speed_total, 20.0) / 20.0,
            min(past_self_error, 20.0) / 20.0,
            min(base_endpoint_disp, 100.0) / 100.0,
            horizon / max(future_len, 1),
            is_traffic,
            is_ped,
            coord_metric,
        ],
        dtype=np.float64,
    )
    return np.concatenate([base, interaction.astype(np.float64)])


def past_baseline_self_error(hist: np.ndarray, dt: float, baseline_name: str) -> float:
    if hist.shape[0] < 6:
        return 0.0
    cut = hist.shape[0] // 2
    prefix = hist[:cut]
    target = hist[cut:]
    pred = rollout(prefix, target.shape[0], dt, baseline_name)[1:]
    return float(np.linalg.norm(pred[:, :, 0:2] - target[:, :, 0:2], axis=2).mean())


def collect_examples(split: str, baselines: Dict) -> List[Dict]:
    hard = hard_lookup()
    encoders: Dict[str, Stage5B6InteractionEncoder] = {}
    examples = []
    for dataset, row in baselines.get("datasets", {}).items():
        baseline_name = row["strongest_causal_baseline"]
        encoders[dataset] = Stage5B6InteractionEncoder(dataset)
        for ep in load_dataset_episodes(dataset, split=split):
            states = ep["states"]
            meta = ep["meta"]
            past = int(meta.get("past_horizon", 10))
            future_len = states.shape[0] - past
            if future_len <= 0:
                continue
            dt = float(meta.get("dt_s", 1.0))
            base = rollout(states[:past], future_len, dt, baseline_name)[1:]
            true = states[past:]
            episode_id = int(meta.get("episode_id", -1))
            hrow = hard.get((dataset, episode_id), {})
            hardness = hrow.get("hardness", "easy")
            interaction = encoders[dataset].encode_episode(meta).as_array()
            for horizon in [h for h in [1, 10, 25, 50, 100] if h <= future_len]:
                baseline_fde = float(np.linalg.norm(base[horizon - 1, :, 0:2] - true[horizon - 1, :, 0:2], axis=1).mean())
                failure = baseline_fde > threshold_for(dataset, horizon, subset=hardness)
                examples.append(
                    {
                        "dataset": dataset,
                        "episode_id": episode_id,
                        "split": split,
                        "horizon": horizon,
                        "future_len": future_len,
                        "hardness": hardness,
                        "events": hrow.get("events", []),
                        "x": stage5b6_feature_vector(dataset, states[:past], meta, baseline_name, horizon, future_len, interaction),
                        "y": (true[horizon - 1, 0, 0:2] - base[horizon - 1, 0, 0:2]).astype(np.float64),
                        "baseline_fde": baseline_fde,
                        "baseline_failure": bool(failure),
                        "baseline_name": baseline_name,
                    }
                )
    return examples


def example_weight(example: Dict, variant_cfg: Dict) -> float:
    w = 1.0
    if example["hardness"] == "hard":
        w *= float(variant_cfg.get("hard_weight", 1.0))
    elif example["hardness"] == "medium":
        w *= 1.35
    if example["baseline_failure"]:
        w *= float(variant_cfg.get("failure_weight", 1.0))
    return float(w)


def choose_alpha_scale(head: Dict, val_rows: List[Dict], feature_mode: str) -> Tuple[float, float]:
    if not val_rows:
        return 0.0, 0.0
    coef = np.asarray(head["residual_coef"], dtype=np.float64)
    failure_coef = np.asarray(head["failure_coef"], dtype=np.float64)
    mean = np.asarray(head["x_mean"], dtype=np.float64)
    scale = np.asarray(head["x_scale"], dtype=np.float64)
    best = (0.0, 0.0, float("inf"))
    for alpha_scale in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for alpha_bias in [0.0, 0.05]:
            losses = []
            for row in val_rows:
                x = trim_features_for_mode(row["x"], feature_mode)
                xz = (x - mean) / np.maximum(scale, 1e-6)
                xb = np.concatenate([[1.0], xz])
                residual = np.tanh((xb @ coef) / 4.0) * 4.0
                prob = 1.0 / (1.0 + np.exp(-np.clip(xb @ failure_coef, -30.0, 30.0)))
                alpha = np.clip(alpha_scale * prob + alpha_bias, 0.0, 1.0)
                penalty = 0.02 * alpha if row["hardness"] == "easy" else 0.0
                losses.append(float(np.linalg.norm(alpha * residual - row["y"]) + penalty))
            score = float(np.mean(losses))
            if score < best[2]:
                best = (alpha_scale, alpha_bias, score)
    return float(best[0]), float(best[1])


def train_variant(name: str, train_rows: List[Dict], val_rows: List[Dict], cfg: Dict) -> Dict:
    heads = {}
    feature_mode = cfg.get("feature_mode", "full")
    for key in sorted({(r["dataset"], r["horizon"]) for r in train_rows}):
        rows = [r for r in train_rows if (r["dataset"], r["horizon"]) == key]
        if not rows:
            continue
        x_raw = np.stack([trim_features_for_mode(r["x"], feature_mode) for r in rows])
        mean = x_raw.mean(axis=0)
        scale = x_raw.std(axis=0)
        x = (x_raw - mean) / np.maximum(scale, 1e-6)
        y = np.stack([r["y"] for r in rows])
        weights = np.asarray([example_weight(r, cfg) for r in rows], dtype=np.float64)
        failures = np.asarray([float(r["baseline_failure"]) for r in rows], dtype=np.float64)
        residual_coef = fit_weighted_ridge(x, y, weights, ridge=2e-2)
        failure_coef = fit_logistic(x, failures, weights, steps=400, lr=0.08, l2=2e-3)
        head = {
            "dataset": key[0],
            "horizon": key[1],
            "feature_mode": feature_mode,
            "x_mean": mean.tolist(),
            "x_scale": scale.tolist(),
            "residual_coef": residual_coef.tolist(),
            "failure_coef": failure_coef.tolist(),
            "alpha_scale": 1.0,
            "alpha_bias": 0.0,
        }
        matching_val = [r for r in val_rows if (r["dataset"], r["horizon"]) == key]
        head["alpha_scale"], head["alpha_bias"] = choose_alpha_scale(head, matching_val, feature_mode)
        heads[f"{key[0]}::{key[1]}"] = head
    payload = {
        "model": "stage5b6_baseline_aware_gated_residual",
        "variant": name,
        "latent_enabled": False,
        "smc_enabled": False,
        "residual_clip": 4.0,
        "heads": heads,
    }
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"variant": name, "checkpoint": str(path), "heads": len(heads), "nonzero_alpha_heads": sum(1 for h in heads.values() if h["alpha_scale"] > 0 or h["alpha_bias"] > 0)}


def train_all_variants() -> Dict:
    baselines = load_json(REPORT_DIR / "stage5b_baseline_metrics.json", {"datasets": {}})
    train = collect_examples("train", baselines)
    val = collect_examples("val", baselines)
    results = [train_variant(name, train, val, cfg) for name, cfg in VARIANTS.items()]
    payload = {"stage": "5B.6", "variants": results, "train_examples": len(train), "val_examples": len(val)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage5b6_gated_residual_training.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

