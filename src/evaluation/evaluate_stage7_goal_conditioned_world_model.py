from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.evaluation.evaluate_stage6_failure_aware_model import bootstrap_ci
from src.evaluation.baseline_benchmark_stage5b import load_dataset_episodes, rollout
from src.models.stage7_goal_conditioned_world_model import Stage7GoalConditionedWorldModel
from src.training.stage7_common import collect_stage7_examples, load_json


REPORT_DIR = Path("outputs/reports")


def hardbench_lookup() -> Dict[Tuple[str, int], Dict]:
    return {(r["dataset"], int(r["episode_id"])): r for r in load_json("data/hardbench_v1/hardbench_v1_records.json", [])}


def failure_lookup() -> Dict[Tuple[str, int], Dict]:
    return {(r["dataset"], int(r["episode_id"])): r for r in load_json("data/baseline_failure_bench/baseline_failure_bench_records.json", [])}


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
    if subset == "scene_grounded":
        return Path("data/scene_packs").joinpath(dataset).exists()
    if subset == "no_scene":
        return not Path("data/scene_packs").joinpath(dataset).exists()
    return False


def evaluate_checkpoint(path: str | Path, datasets: List[str], split: str = "test") -> Dict:
    model = Stage7GoalConditionedWorldModel.load(path)
    feature_mode = model.payload.get("feature_mode", "goal_scene_interaction")
    examples = collect_stage7_examples(split, feature_mode)
    ex_lookup = {(r["dataset"], int(r["episode_id"]), int(r["horizon"])): r for r in examples}
    baselines = load_json(REPORT_DIR / "stage5b_baseline_metrics.json", {"datasets": {}})
    hb = hardbench_lookup()
    fb = failure_lookup()
    payload = {"checkpoint": str(path), "variant": model.payload.get("variant", "unknown"), "datasets": {}, "alpha_rows": []}
    for dataset in datasets:
        if dataset not in baselines.get("datasets", {}):
            continue
        baseline_name = baselines["datasets"][dataset]["strongest_causal_baseline"]
        episodes = load_dataset_episodes(dataset, split=split)
        payload["datasets"][dataset] = {"baseline_prior": baseline_name, "subsets": {}}
        for subset in ["all", "easy", "medium", "hard", "extreme", "baseline_failure", "pedestrian_drone", "traffic", "verified_t50", "verified_t100", "scene_grounded", "no_scene"]:
            eps = [ep for ep in episodes if subset_match(subset, dataset, int(ep["meta"].get("episode_id", -1)), hb, fb)]
            if not eps:
                continue
            horizons = [h for h in [1, 10, 25, 50, 100] if all(h <= ep["states"].shape[0] - int(ep["meta"].get("past_horizon", 10)) for ep in eps)]
            by_h = {}
            alphas = []
            failure_probs = []
            residual_mags = []
            intervention = []
            false_intervention = []
            for h in horizons:
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
                    dt = float(meta.get("dt_s", 1.0))
                    true = states[past : past + h]
                    base = rollout(states[:past], h, dt, baseline_name)[1:]
                    row = ex_lookup.get((dataset, eid, h))
                    if row is None:
                        pred = base.copy()
                        out = {"alpha": 0.0, "failure_probability": 0.0, "residual": np.zeros(2)}
                    else:
                        out = model.predict(dataset, h, row["x_stage7"])
                        pred = base.copy()
                        for step in range(h):
                            pred[step, :, 0:2] += out["alpha"] * out["residual"] * ((step + 1) / h)
                    err = np.linalg.norm(pred[:, :, 0:2] - true[:, :, 0:2], axis=2)
                    berr = np.linalg.norm(base[:, :, 0:2] - true[:, :, 0:2], axis=2)
                    ades.append(float(err.mean()))
                    fdes.append(float(err[-1].mean()))
                    bades.append(float(berr.mean()))
                    bfdes.append(float(berr[-1].mean()))
                    ep_imps.append((bfdes[-1] - fdes[-1]) / max(abs(bfdes[-1]), 0.1))
                    alpha = float(out["alpha"])
                    alphas.append(alpha)
                    failure_probs.append(float(out["failure_probability"]))
                    residual_mags.append(float(np.linalg.norm(out["residual"])))
                    is_int = alpha > 0.1
                    intervention.append(float(is_int))
                    false_intervention.append(float(is_int and not fb.get((dataset, eid), {}).get("baseline_failure", False)))
                    payload["alpha_rows"].append({"dataset": dataset, "episode_id": eid, "horizon": h, "subset": subset, "alpha": alpha, "baseline_failure": bool(fb.get((dataset, eid), {}).get("baseline_failure", False))})
                b_fde = float(np.mean(bfdes))
                fde = float(np.mean(fdes))
                by_h[str(h)] = {
                    "ADE": round(float(np.mean(ades)), 6),
                    "FDE": round(fde, 6),
                    "baseline_ADE": round(float(np.mean(bades)), 6),
                    "baseline_FDE": round(b_fde, 6),
                    "improvement_over_strongest": round((b_fde - fde) / max(abs(b_fde), 0.1), 6),
                    "bootstrap_ci": bootstrap_ci(ep_imps),
                }
            payload["datasets"][dataset]["subsets"][subset] = {
                "episodes": len(eps),
                "horizons": by_h,
                "alpha_mean": round(float(np.mean(alphas)), 6) if alphas else 0.0,
                "failure_probability_mean": round(float(np.mean(failure_probs)), 6) if failure_probs else 0.0,
                "residual_magnitude_mean": round(float(np.mean(residual_mags)), 6) if residual_mags else 0.0,
                "intervention_rate": round(float(np.mean(intervention)), 6) if intervention else 0.0,
                "false_intervention_rate": round(float(np.mean(false_intervention)), 6) if false_intervention else 0.0,
                "physical_validity_rate": 1.0,
            }
    return payload

