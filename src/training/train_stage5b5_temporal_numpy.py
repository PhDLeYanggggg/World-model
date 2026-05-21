from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.evaluation.baseline_benchmark_stage5b import load_dataset_episodes, rollout


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def hard_lookup() -> Dict[Tuple[str, int], Dict]:
    rows = load_json("outputs/reports/stage5b5_hard_subset_summary.json", [])
    out = {}
    for dataset in rows:
        for ep in dataset.get("episodes", []):
            out[(dataset["dataset_name"], int(ep["episode_id"]))] = ep
    return out


def feature_vector(dataset: str, hist: np.ndarray, hard: Dict, horizon: int, future_len: int) -> np.ndarray:
    last = hist[-1, 0]
    prev = hist[-2, 0] if hist.shape[0] >= 2 else last
    heading_delta = np.angle(np.exp(1j * (last[6] - prev[6])))
    speed_change = float(last[7] - prev[7])
    is_traffic = 1.0 if dataset.startswith("tgsim") else 0.0
    is_ped = 1.0 if dataset in {"trajnet", "eth_ucy"} else 0.0
    return np.asarray(
        [
            1.0,
            last[2],
            last[3],
            last[4],
            last[5],
            last[7],
            np.sin(last[6]),
            np.cos(last[6]),
            heading_delta,
            speed_change,
            min(hard.get("nearest_neighbor_distance_min", 999.0), 50.0) / 50.0,
            min(hard.get("time_to_collision_min", 999.0), 50.0) / 50.0,
            hard.get("interaction_density", 0.0),
            min(max(hard.get("closing_speed", 0.0), -20.0), 20.0) / 20.0,
            is_traffic,
            is_ped,
            horizon / max(future_len, 1),
        ],
        dtype=np.float64,
    )


def collect_examples(split: str, baselines: Dict):
    hard = hard_lookup()
    examples = []
    for dataset, row in baselines["datasets"].items():
        baseline = row["strongest_causal_baseline"]
        for ep in load_dataset_episodes(dataset, split=split):
            states = ep["states"]
            meta = ep["meta"]
            past = int(meta.get("past_horizon", 10))
            future_len = states.shape[0] - past
            dt = float(meta.get("dt_s", 1.0))
            base = rollout(states[:past], future_len, dt, baseline)[1:]
            true = states[past:]
            hard_row = hard.get((dataset, int(meta.get("episode_id", -1))), {})
            weight = 2.0 if hard_row.get("hardness") == "hard" else (1.3 if hard_row.get("hardness") == "medium" else 1.0)
            for horizon in [h for h in [1, 10, 25, 50, 100] if h <= future_len]:
                examples.append(
                    {
                        "dataset": dataset,
                        "episode_id": int(meta.get("episode_id", -1)),
                        "horizon": horizon,
                        "future_len": future_len,
                        "x": feature_vector(dataset, states[:past], hard_row, horizon, future_len),
                        "y": (true[horizon - 1, 0, 0:2] - base[horizon - 1, 0, 0:2]).astype(np.float64),
                        "weight": weight,
                        "baseline": baseline,
                    }
                )
    return examples


def fit_ridge(examples: List[Dict], ridge: float = 1e-2) -> Dict:
    heads = {}
    for key in sorted({(e["dataset"], e["horizon"]) for e in examples}):
        rows = [e for e in examples if (e["dataset"], e["horizon"]) == key]
        x = np.stack([e["x"] for e in rows])
        y = np.stack([e["y"] for e in rows])
        w = np.sqrt(np.asarray([e["weight"] for e in rows]))[:, None]
        xw = x * w
        yw = y * w
        coef = np.linalg.solve(xw.T @ xw + ridge * np.eye(x.shape[1]), xw.T @ yw)
        heads[f"{key[0]}::{key[1]}"] = {"coef": coef.tolist(), "alpha": 1.0}
    return heads


def choose_alpha(heads: Dict, val_examples: List[Dict], baselines: Dict) -> Dict:
    # Validation-only stability gate; if residual hurts validation, shrink it toward baseline.
    for key, head in heads.items():
        dataset, horizon_s = key.split("::")
        horizon = int(horizon_s)
        rows = [e for e in val_examples if e["dataset"] == dataset and e["horizon"] == horizon]
        if not rows:
            head["alpha"] = 0.0
            continue
        coef = np.asarray(head["coef"])
        best_alpha, best_err = 0.0, float("inf")
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            errs = []
            for e in rows:
                pred_res = e["x"] @ coef * alpha
                errs.append(float(np.linalg.norm(pred_res - e["y"])))
            score = float(np.mean(errs))
            if score < best_err:
                best_err = score
                best_alpha = alpha
        head["alpha"] = best_alpha
    return heads


def train_numpy_temporal() -> Dict:
    baselines = load_json("outputs/reports/stage5b_baseline_metrics.json", {"datasets": {}})
    train = collect_examples("train", baselines)
    val = collect_examples("val", baselines)
    heads = choose_alpha(fit_ridge(train), val, baselines)
    payload = {
        "model": "numpy_temporal_interaction_ridge_residual",
        "latent_enabled": False,
        "smc_enabled": False,
        "heads": heads,
    }
    out = Path("outputs/checkpoints/stage5b5/temporal_interaction_numpy.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"checkpoint": str(out), "heads": len(heads), "nonzero_alpha_heads": sum(1 for h in heads.values() if h["alpha"] > 0)}
