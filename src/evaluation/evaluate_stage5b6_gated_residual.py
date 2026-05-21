from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.evaluation.baseline_benchmark_stage5b import load_dataset_episodes, rollout
from src.models.encoders.stage5b6_interaction_encoder import Stage5B6InteractionEncoder
from src.models.stage5b6_gated_residual_model import Stage5B6GatedResidualModel
from src.training.train_stage5b6_gated_residual import stage5b6_feature_vector


REPORT_DIR = Path("outputs/reports")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def hard_lookup() -> Dict[Tuple[str, int], Dict]:
    rows = load_json(REPORT_DIR / "stage5b5_hard_subset_summary.json", [])
    out = {}
    for dataset in rows:
        scores = [float(ep.get("hard_score", 0.0)) for ep in dataset.get("episodes", [])]
        extreme_cut = float(np.quantile(scores, 0.9)) if scores else 1.0
        for ep in dataset.get("episodes", []):
            ep = dict(ep)
            ep["is_extreme"] = float(ep.get("hard_score", 0.0)) >= extreme_cut
            out[(dataset["dataset_name"], int(ep["episode_id"]))] = ep
    return out


def subset_match(subset: str, hrow: Dict) -> bool:
    if subset == "all":
        return True
    if subset == "extreme":
        return bool(hrow.get("is_extreme", False))
    return hrow.get("hardness", "easy") == subset


def apply_prediction(states: np.ndarray, meta: Dict, dataset: str, horizon: int, baseline_name: str, model: Stage5B6GatedResidualModel, interaction_encoder: Stage5B6InteractionEncoder, use_interaction: bool = True):
    past = int(meta.get("past_horizon", 10))
    future_len = states.shape[0] - past
    dt = float(meta.get("dt_s", 1.0))
    base = rollout(states[:past], horizon, dt, baseline_name)[1:]
    interaction = interaction_encoder.encode_episode(meta).as_array()
    x = stage5b6_feature_vector(dataset, states[:past], meta, baseline_name, horizon, future_len, interaction)
    pred_res = model.predict(dataset, horizon, x, use_interaction=use_interaction)
    pred = base.copy()
    for step in range(horizon):
        pred[step, :, 0:2] += pred_res.alpha * pred_res.residual * ((step + 1) / horizon)
    return base, pred, pred_res


def evaluate_checkpoint(checkpoint: str | Path, datasets: List[str], split: str = "test", use_interaction: bool = True) -> Dict:
    model = Stage5B6GatedResidualModel.load(checkpoint)
    baselines = load_json(REPORT_DIR / "stage5b_baseline_metrics.json", {"datasets": {}})
    hard = hard_lookup()
    payload = {"checkpoint": str(checkpoint), "variant": model.payload.get("variant", "unknown"), "datasets": {}, "alpha_rows": []}
    for dataset in datasets:
        baseline_name = baselines["datasets"][dataset]["strongest_causal_baseline"]
        interaction_encoder = Stage5B6InteractionEncoder(dataset)
        episodes = load_dataset_episodes(dataset, split=split)
        payload["datasets"][dataset] = {"baseline_prior": baseline_name, "subsets": {}}
        for subset in ["all", "easy", "medium", "hard", "extreme"]:
            subset_eps = []
            for ep in episodes:
                episode_id = int(ep["meta"].get("episode_id", -1))
                hrow = hard.get((dataset, episode_id), {})
                if subset_match(subset, hrow):
                    subset_eps.append(ep)
            if not subset_eps:
                continue
            horizons = [h for h in [1, 10, 25, 50, 100] if all(h <= ep["states"].shape[0] - int(ep["meta"].get("past_horizon", 10)) for ep in subset_eps)]
            by_h = {}
            alphas = []
            residual_mags = []
            failure_probs = []
            episode_improvements_by_target = []
            speed_violations = []
            accel_violations = []
            for horizon in horizons:
                ades = []
                fdes = []
                bades = []
                bfdes = []
                ep_improvements = []
                for ep in subset_eps:
                    states = ep["states"]
                    meta = ep["meta"]
                    past = int(meta.get("past_horizon", 10))
                    true = states[past : past + horizon]
                    base, pred, pred_res = apply_prediction(states, meta, dataset, horizon, baseline_name, model, interaction_encoder, use_interaction=use_interaction)
                    err = np.linalg.norm(pred[:, :, 0:2] - true[:, :, 0:2], axis=2)
                    berr = np.linalg.norm(base[:, :, 0:2] - true[:, :, 0:2], axis=2)
                    ades.append(float(err.mean()))
                    fdes.append(float(err[-1].mean()))
                    bades.append(float(berr.mean()))
                    bfdes.append(float(berr[-1].mean()))
                    ep_improvements.append((bfdes[-1] - fdes[-1]) / max(abs(bfdes[-1]), 0.1))
                    alphas.append(pred_res.alpha)
                    residual_mags.append(float(np.linalg.norm(pred_res.residual)))
                    failure_probs.append(pred_res.failure_probability)
                    payload["alpha_rows"].append(
                        {
                            "dataset": dataset,
                            "episode_id": int(meta.get("episode_id", -1)),
                            "horizon": horizon,
                            "hardness": hard.get((dataset, int(meta.get("episode_id", -1))), {}).get("hardness", "easy"),
                            "alpha": pred_res.alpha,
                            "failure_probability": pred_res.failure_probability,
                        }
                    )
                    speed = np.linalg.norm(pred[:, :, 2:4], axis=2)
                    accel = np.linalg.norm(np.diff(pred[:, :, 2:4], axis=0) / max(float(meta.get("dt_s", 1.0)), 1e-6), axis=2) if pred.shape[0] > 1 else np.zeros_like(speed)
                    true_speed = np.linalg.norm(true[:, :, 2:4], axis=2)
                    speed_limit = max(1.0, float(np.nanpercentile(true_speed, 99.0)) * 2.0)
                    accel_limit = max(1.0, float(np.nanpercentile(np.linalg.norm(true[:, :, 4:6], axis=2), 99.0)) * 3.0)
                    speed_violations.append(float(np.mean(speed > speed_limit)))
                    accel_violations.append(float(np.mean(accel > accel_limit)) if accel.size else 0.0)
                b_fde = float(np.mean(bfdes))
                l_fde = float(np.mean(fdes))
                ci = bootstrap_ci(ep_improvements)
                by_h[str(horizon)] = {
                    "ADE": round(float(np.mean(ades)), 6),
                    "FDE": round(l_fde, 6),
                    "baseline_ADE": round(float(np.mean(bades)), 6),
                    "baseline_FDE": round(b_fde, 6),
                    "improvement_over_strongest": round((b_fde - l_fde) / max(abs(b_fde), 1e-9), 6),
                    "bootstrap_ci": ci,
                }
                if horizon == (100 if "100" in [str(x) for x in horizons] else max(horizons)):
                    episode_improvements_by_target = ep_improvements
            payload["datasets"][dataset]["subsets"][subset] = {
                "episodes": len(subset_eps),
                "horizons": by_h,
                "alpha_mean": round(float(np.mean(alphas)), 6) if alphas else 0.0,
                "failure_probability_mean": round(float(np.mean(failure_probs)), 6) if failure_probs else 0.0,
                "residual_magnitude_mean": round(float(np.mean(residual_mags)), 6) if residual_mags else 0.0,
                "physical_validity_rate": round(float(max(0.0, 1.0 - np.mean(speed_violations + accel_violations))), 6) if speed_violations else 1.0,
                "speed_violation_rate": round(float(np.mean(speed_violations)), 6) if speed_violations else 0.0,
                "acceleration_violation_rate": round(float(np.mean(accel_violations)), 6) if accel_violations else 0.0,
                "target_improvement_ci": bootstrap_ci(episode_improvements_by_target),
            }
    return payload


def bootstrap_ci(values: List[float], samples: int = 500, seed: int = 11) -> Dict:
    if not values:
        return {"mean": 0.0, "ci_low": None, "ci_high": None, "n": 0}
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return {"mean": round(float(vals.mean()), 6), "ci_low": None, "ci_high": None, "n": int(len(vals))}
    rng = np.random.default_rng(seed)
    means = [float(vals[rng.integers(0, len(vals), size=len(vals))].mean()) for _ in range(samples)]
    return {"mean": round(float(vals.mean()), 6), "ci_low": round(float(np.quantile(means, 0.025)), 6), "ci_high": round(float(np.quantile(means, 0.975)), 6), "n": int(len(vals))}
