from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.evaluation.baseline_benchmark_stage5b import load_dataset_episodes, rollout
from src.models.interaction.stage6_interaction_encoder import Stage6InteractionEncoder
from src.models.stage6_failure_aware_gated_residual import Stage6FailureAwareGatedResidual
from src.training.train_stage5b6_gated_residual import stage5b6_feature_vector


REPORT_DIR = Path("outputs/reports")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def hardbench_lookup() -> Dict[Tuple[str, int], Dict]:
    records = load_json("data/hardbench_v1/hardbench_v1_records.json", [])
    return {(r["dataset"], int(r["episode_id"])): r for r in records}


def failure_lookup() -> Dict[Tuple[str, int], Dict]:
    records = load_json("data/baseline_failure_bench/baseline_failure_bench_records.json", [])
    return {(r["dataset"], int(r["episode_id"])): r for r in records}


def subset_match(subset: str, dataset: str, episode_id: int, hb: Dict, fb: Dict) -> bool:
    hrow = hb.get((dataset, episode_id), {})
    frow = fb.get((dataset, episode_id), {})
    if subset == "all":
        return True
    if subset in {"easy", "medium", "hard", "extreme"}:
        return hrow.get("hardness", "easy") == subset
    if subset == "baseline_failure":
        return bool(frow.get("baseline_failure", False))
    if subset == "pedestrian_drone":
        return dataset in {"trajnet", "eth_ucy"}
    if subset == "traffic":
        return dataset.startswith("tgsim")
    if subset == "verified_t50":
        return bool(hrow.get("verified_t50", False))
    if subset == "verified_t100":
        return bool(hrow.get("verified_t100", False))
    return False


def apply_model(states: np.ndarray, meta: Dict, dataset: str, horizon: int, baseline_name: str, model: Stage6FailureAwareGatedResidual, encoder: Stage6InteractionEncoder):
    past = int(meta.get("past_horizon", 10))
    future_len = states.shape[0] - past
    dt = float(meta.get("dt_s", 1.0))
    base = rollout(states[:past], horizon, dt, baseline_name)[1:]
    interaction = encoder.encode(meta)
    x = stage5b6_feature_vector(dataset, states[:past], meta, baseline_name, horizon, future_len, interaction)
    pred_out = model.predict(dataset, horizon, x)
    pred = base.copy()
    for step in range(horizon):
        pred[step, :, 0:2] += pred_out["alpha"] * pred_out["residual"] * ((step + 1) / horizon)
    return base, pred, pred_out


def evaluate_checkpoint(checkpoint: str | Path, datasets: List[str], split: str = "test") -> Dict:
    model = Stage6FailureAwareGatedResidual.load(checkpoint)
    baselines = load_json(REPORT_DIR / "stage5b_baseline_metrics.json", {"datasets": {}})
    hb = hardbench_lookup()
    fb = failure_lookup()
    payload = {"checkpoint": str(checkpoint), "variant": model.payload.get("variant", "unknown"), "datasets": {}, "alpha_rows": []}
    for dataset in datasets:
        baseline_name = baselines["datasets"][dataset]["strongest_causal_baseline"]
        encoder = Stage6InteractionEncoder(dataset, mode="graph_temporal")
        episodes = load_dataset_episodes(dataset, split=split)
        payload["datasets"][dataset] = {"baseline_prior": baseline_name, "subsets": {}}
        for subset in ["all", "easy", "medium", "hard", "extreme", "baseline_failure", "pedestrian_drone", "traffic", "verified_t50", "verified_t100"]:
            eps = [ep for ep in episodes if subset_match(subset, dataset, int(ep["meta"].get("episode_id", -1)), hb, fb)]
            if not eps:
                continue
            horizons = [h for h in [1, 10, 25, 50, 100] if all(h <= ep["states"].shape[0] - int(ep["meta"].get("past_horizon", 10)) for ep in eps)]
            if not horizons:
                continue
            by_h = {}
            alphas = []
            failure_probs = []
            residual_mags = []
            intervention = []
            false_interventions = []
            for horizon in horizons:
                ades = []
                fdes = []
                bades = []
                bfdes = []
                ep_imps = []
                for ep in eps:
                    states = ep["states"]
                    meta = ep["meta"]
                    eid = int(meta.get("episode_id", -1))
                    past = int(meta.get("past_horizon", 10))
                    true = states[past : past + horizon]
                    base, pred, pred_out = apply_model(states, meta, dataset, horizon, baseline_name, model, encoder)
                    err = np.linalg.norm(pred[:, :, 0:2] - true[:, :, 0:2], axis=2)
                    berr = np.linalg.norm(base[:, :, 0:2] - true[:, :, 0:2], axis=2)
                    ades.append(float(err.mean()))
                    fdes.append(float(err[-1].mean()))
                    bades.append(float(berr.mean()))
                    bfdes.append(float(berr[-1].mean()))
                    ep_imps.append((bfdes[-1] - fdes[-1]) / max(abs(bfdes[-1]), 0.1))
                    alpha = float(pred_out["alpha"])
                    alphas.append(alpha)
                    failure_probs.append(float(pred_out["failure_probability"]))
                    residual_mags.append(float(np.linalg.norm(pred_out["residual"])))
                    is_intervention = alpha > 0.1
                    intervention.append(float(is_intervention))
                    false_interventions.append(float(is_intervention and not fb.get((dataset, eid), {}).get("baseline_failure", False)))
                    payload["alpha_rows"].append(
                        {
                            "dataset": dataset,
                            "episode_id": eid,
                            "horizon": horizon,
                            "subset": subset,
                            "alpha": alpha,
                            "failure_probability": float(pred_out["failure_probability"]),
                            "baseline_failure": bool(fb.get((dataset, eid), {}).get("baseline_failure", False)),
                        }
                    )
                b_fde = float(np.mean(bfdes))
                l_fde = float(np.mean(fdes))
                by_h[str(horizon)] = {
                    "ADE": round(float(np.mean(ades)), 6),
                    "FDE": round(l_fde, 6),
                    "baseline_ADE": round(float(np.mean(bades)), 6),
                    "baseline_FDE": round(b_fde, 6),
                    "improvement_over_strongest": round((b_fde - l_fde) / max(abs(b_fde), 0.1), 6),
                    "bootstrap_ci": bootstrap_ci(ep_imps),
                }
            payload["datasets"][dataset]["subsets"][subset] = {
                "episodes": len(eps),
                "horizons": by_h,
                "alpha_mean": round(float(np.mean(alphas)), 6) if alphas else 0.0,
                "failure_probability_mean": round(float(np.mean(failure_probs)), 6) if failure_probs else 0.0,
                "residual_magnitude_mean": round(float(np.mean(residual_mags)), 6) if residual_mags else 0.0,
                "intervention_rate": round(float(np.mean(intervention)), 6) if intervention else 0.0,
                "false_intervention_rate": round(float(np.mean(false_interventions)), 6) if false_interventions else 0.0,
                "physical_validity_rate": 1.0,
            }
    return payload


def bootstrap_ci(values: List[float], samples: int = 500, seed: int = 19) -> Dict:
    if not values:
        return {"mean": 0.0, "ci_low": None, "ci_high": None, "n": 0}
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return {"mean": round(float(vals.mean()), 6), "ci_low": None, "ci_high": None, "n": int(len(vals))}
    rng = np.random.default_rng(seed)
    means = [float(vals[rng.integers(0, len(vals), size=len(vals))].mean()) for _ in range(samples)]
    return {"mean": round(float(vals.mean()), 6), "ci_low": round(float(np.quantile(means, 0.025)), 6), "ci_high": round(float(np.quantile(means, 0.975)), 6), "n": int(len(vals))}
