from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.evaluation.stage9_data_audit import load_stage9_episodes
from src.evaluation.stage9_per_agent_baselines import masked_ade_fde, physical_validity, rollout_baseline
from src.models.stage9_per_agent_world_model import Stage9PerAgentWorldModel
from src.training.train_stage9_per_agent import select_features, stage9_feature_vector


REPORT_DIR = Path("outputs/reports")


def load_baselines() -> Dict:
    p = REPORT_DIR / "stage9_per_agent_baseline_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"datasets": {}}


def predict_episode(model: Stage9PerAgentWorldModel, ep: Dict, horizon: int, baseline_name: str) -> Tuple[np.ndarray, Dict]:
    base = rollout_baseline(ep, horizon, baseline_name)
    pred = base.copy()
    mode = model.payload.get("feature_mode", "full")
    alphas = []
    failures = []
    residuals = []
    past = int(ep["meta"]["past_horizon"])
    valid_agents = np.where(ep["mask"][past - 1] & ep["mask"][past + horizon - 1])[0]
    for agent_idx in valid_agents:
        x_full = stage9_feature_vector(ep, base, int(agent_idx), horizon)
        out = model.predict_residual(ep["meta"]["dataset_name"], horizon, select_features(x_full, mode))
        for t in range(horizon):
            pred[t, agent_idx, 0:2] += out["alpha"] * out["residual"] * ((t + 1) / horizon)
        alphas.append(float(out["alpha"]))
        failures.append(float(out["failure_probability"]))
        residuals.append(float(np.linalg.norm(out["residual"])))
    diag = {
        "alpha_mean": float(np.mean(alphas)) if alphas else 0.0,
        "failure_probability_mean": float(np.mean(failures)) if failures else 0.0,
        "residual_magnitude_mean": float(np.mean(residuals)) if residuals else 0.0,
        "intervention_rate": float(np.mean(np.asarray(alphas) > 0.1)) if alphas else 0.0,
    }
    return pred, diag


def evaluate_checkpoint(path: str | Path, split: str = "test") -> Dict:
    model = Stage9PerAgentWorldModel.load(path)
    baselines = load_baselines()
    payload = {"checkpoint": str(path), "variant": model.payload.get("variant"), "datasets": {}}
    for dataset, drow in baselines.get("datasets", {}).items():
        episodes = load_stage9_episodes(dataset, split=split)
        if not episodes and split == "test":
            episodes = load_stage9_episodes(dataset, split="val")
        baseline_name = drow["strongest_causal_baseline"]
        payload["datasets"][dataset] = {"baseline": baseline_name, "subsets": {}}
        for subset in subsets():
            eps = [ep for ep in episodes if subset_match(ep, subset)]
            if not eps:
                continue
            horizons = [h for h in [1, 5, 10, 25, 50, 100] if all(h <= int(ep["meta"]["future_horizon"]) for ep in eps)]
            if not horizons:
                continue
            payload["datasets"][dataset]["subsets"][subset] = evaluate_subset(model, eps, horizons, baseline_name)
    return payload


def subsets() -> List[str]:
    return ["all", "easy", "hard", "baseline_failure", "goalbench_official", "silver", "inferred_only", "ge2", "ge5", "pedestrian_drone", "traffic", "pixel_space", "metric"]


def subset_match(ep: Dict, subset: str) -> bool:
    meta = ep["meta"]
    if subset == "all":
        return True
    if subset == "easy":
        return not meta.get("hard_interaction") and not meta.get("baseline_failure_proxy")
    if subset == "hard":
        return bool(meta.get("hard_interaction"))
    if subset == "baseline_failure":
        return bool(meta.get("baseline_failure_proxy"))
    if subset == "goalbench_official":
        return meta.get("annotation_quality") in {"gold", "silver"}
    if subset == "silver":
        return meta.get("annotation_quality") == "silver"
    if subset == "inferred_only":
        return meta.get("annotation_quality") == "inferred_only"
    if subset == "ge2":
        return int(meta.get("agent_count", 0)) >= 2
    if subset == "ge5":
        return int(meta.get("agent_count", 0)) >= 5
    if subset == "pedestrian_drone":
        return meta.get("dataset_name") in {"trajnet", "eth_ucy", "sdd", "opentraj", "aerialmpt_long"}
    if subset == "traffic":
        return str(meta.get("dataset_name", "")).startswith("tgsim")
    if subset == "pixel_space":
        return meta.get("coordinate_unit") == "pixel"
    if subset == "metric":
        return meta.get("coordinate_unit") == "meter"
    return False


def evaluate_subset(model: Stage9PerAgentWorldModel, episodes: List[Dict], horizons: List[int], baseline_name: str) -> Dict:
    out = {"episodes": len(episodes), "horizons": {}}
    alpha_rows = []
    for horizon in horizons:
        ades = []
        fdes = []
        bades = []
        bfdes = []
        validity = []
        imps = []
        for ep in episodes:
            past = int(ep["meta"]["past_horizon"])
            true = ep["states"][past : past + horizon]
            mask = ep["mask"][past : past + horizon]
            base = rollout_baseline(ep, horizon, baseline_name)
            pred, diag = predict_episode(model, ep, horizon, baseline_name)
            ade, fde = masked_ade_fde(pred, true, mask)
            bade, bfde = masked_ade_fde(base, true, mask)
            ades.append(ade)
            fdes.append(fde)
            bades.append(bade)
            bfdes.append(bfde)
            validity.append(physical_validity(pred, mask))
            imps.append((bfde - fde) / max(abs(bfde), 0.1) if not math.isnan(fde) and not math.isnan(bfde) else 0.0)
            alpha_rows.append(diag)
        b_fde = float(np.nanmean(bfdes))
        fde = float(np.nanmean(fdes))
        out["horizons"][str(horizon)] = {
            "ADE": round(float(np.nanmean(ades)), 6),
            "FDE": round(fde, 6),
            "baseline_ADE": round(float(np.nanmean(bades)), 6),
            "baseline_FDE": round(b_fde, 6),
            "improvement_over_strongest": round((b_fde - fde) / max(abs(b_fde), 0.1), 6),
            "bootstrap_ci": bootstrap_ci(imps),
            "physical_validity": round(float(np.nanmean(validity)), 6),
        }
    out.update(
        {
            "alpha_mean": round(float(np.mean([r["alpha_mean"] for r in alpha_rows])), 6) if alpha_rows else 0.0,
            "failure_probability_mean": round(float(np.mean([r["failure_probability_mean"] for r in alpha_rows])), 6) if alpha_rows else 0.0,
            "residual_magnitude_mean": round(float(np.mean([r["residual_magnitude_mean"] for r in alpha_rows])), 6) if alpha_rows else 0.0,
            "intervention_rate": round(float(np.mean([r["intervention_rate"] for r in alpha_rows])), 6) if alpha_rows else 0.0,
        }
    )
    return out


def bootstrap_ci(values: List[float]) -> Dict:
    if not values:
        return {"ci_low": 0.0, "ci_high": 0.0}
    arr = np.asarray(values, dtype=float)
    if len(arr) < 3:
        return {"ci_low": round(float(arr.mean()), 6), "ci_high": round(float(arr.mean()), 6)}
    rng = np.random.default_rng(9)
    samples = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(200)]
    return {"ci_low": round(float(np.percentile(samples, 2.5)), 6), "ci_high": round(float(np.percentile(samples, 97.5)), 6)}
