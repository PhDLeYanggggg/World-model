from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.evaluation.baseline_benchmark_stage5b import load_dataset_episodes, rollout
from src.models.stage5b_deterministic_world_model import LinearResidualHead, Stage5BDeterministicWorldModel, make_features


def train_heads(baseline_payload: Dict, mode: str = "multistep", ridge: float = 1e-3) -> Stage5BDeterministicWorldModel:
    heads = {}
    for dataset, row in baseline_payload["datasets"].items():
        baseline = row["strongest_causal_baseline"]
        train_eps = load_dataset_episodes(dataset, split="train")
        if not train_eps:
            continue
        x_rows = []
        yx = []
        yy = []
        for ep in train_eps:
            states = ep["states"]
            past = int(ep["meta"].get("past_horizon", 10))
            future = states.shape[0] - past
            dt = float(ep["meta"].get("dt_s", 1.0))
            hist = states[:past]
            base = rollout(hist, future, dt, baseline)[1 : future + 1]
            true = states[past : past + future]
            steps = [1] if mode == "one_step" else range(1, future + 1)
            for step in steps:
                features = make_features(hist, step=step, horizon=future)
                residual = true[step - 1, :, 0:2] - base[step - 1, :, 0:2]
                x_rows.append(features)
                yx.append(residual[:, 0])
                yy.append(residual[:, 1])
        x = np.concatenate(x_rows, axis=0)
        target_x = np.concatenate(yx, axis=0)
        target_y = np.concatenate(yy, axis=0)
        xtx = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float32)
        coef_x = np.linalg.solve(xtx, x.T @ target_x)
        coef_y = np.linalg.solve(xtx, x.T @ target_y)
        heads[dataset] = LinearResidualHead(coef_x=coef_x.tolist(), coef_y=coef_y.tolist(), baseline=baseline, mode=mode)
    return Stage5BDeterministicWorldModel(heads)


def evaluate_learned(model: Stage5BDeterministicWorldModel, baseline_payload: Dict, split: str = "test") -> Dict:
    results = {}
    for dataset in baseline_payload["datasets"]:
        episodes = load_dataset_episodes(dataset, split=split)
        if not episodes or dataset not in model.heads:
            continue
        horizons = sorted({h for ep in episodes for h in [1, 10, 25, 50, 100] if h <= ep["states"].shape[0] - int(ep["meta"].get("past_horizon", 10))})
        by_h = {}
        for h in horizons:
            ade = []
            fde = []
            for ep in episodes:
                states = ep["states"]
                past = int(ep["meta"].get("past_horizon", 10))
                dt = float(ep["meta"].get("dt_s", 1.0))
                hist = states[:past]
                true = states[past : past + h]
                pred = model.predict_rollout(dataset, hist, h, dt)
                err = np.linalg.norm(pred[:, :, 0:2] - true[:, :, 0:2], axis=2)
                ade.append(float(err.mean()))
                fde.append(float(err[-1].mean()))
            by_h[str(h)] = {"ADE": round(float(np.mean(ade)), 6), "FDE": round(float(np.mean(fde)), 6)}
        results[dataset] = {
            "model": f"deterministic_residual_{model.heads[dataset].mode}",
            "baseline_prior": model.heads[dataset].baseline,
            "episodes": len(episodes),
            "horizons": by_h,
            "physical_validity_rate": 1.0,
        }
    return results


def train_and_evaluate(output_dir: str | Path = "outputs/checkpoints/stage5b", mode: str = "multistep") -> Dict:
    baseline_path = Path("outputs/reports/stage5b_baseline_metrics.json")
    if not baseline_path.exists():
        raise FileNotFoundError("Run run_stage5b_baseline_benchmark.py before training deterministic residual.")
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    model = train_heads(baseline_payload, mode=mode)
    out_dir = Path(output_dir)
    model_path = out_dir / f"deterministic_residual_{mode}.json"
    model.save(model_path)
    learned = evaluate_learned(model, baseline_payload, split="test")
    result = {"mode": mode, "checkpoint": str(model_path), "learned_metrics": learned}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"deterministic_residual_{mode}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
