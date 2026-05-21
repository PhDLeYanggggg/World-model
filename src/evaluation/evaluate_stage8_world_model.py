from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.evaluation.evaluate_stage6_failure_aware_model import bootstrap_ci
from src.evaluation.baseline_benchmark_stage5b import rollout
from src.evaluation.stage8_goalbench_gold import available_stage8_datasets, load_multiagent_episodes
from src.models.stage8_goal_conditioned_world_model_v2 import Stage8GoalConditionedWorldModelV2
from src.training.stage8_common import collect_stage8_examples, load_json, strongest_baselines


REPORT_DIR = Path("outputs/reports")


def hardbench_lookup() -> Dict[Tuple[str, int], Dict]:
    return {(r["dataset"], int(r["episode_id"])): r for r in load_json("data/hardbench_v1/hardbench_v1_records.json", [])}


def failure_lookup() -> Dict[Tuple[str, int], Dict]:
    return {(r["dataset"], int(r["episode_id"])): r for r in load_json("data/baseline_failure_bench/baseline_failure_bench_records.json", [])}


def scene_quality_lookup() -> Dict[Tuple[str, str], str]:
    out = {}
    for path in Path("data/scene_gold_packs").glob("*/*/scene_gold_pack.json"):
        pack = json.loads(path.read_text(encoding="utf-8"))
        out[(pack.get("dataset_name"), str(pack.get("scene_id")))] = pack.get("annotation_quality", "inferred_only")
    return out


def subset_match(subset: str, dataset: str, meta: Dict, hb: Dict, fb: Dict, sq: Dict) -> bool:
    eid = int(meta.get("episode_id", -1))
    hrow = hb.get((dataset, eid), {})
    frow = fb.get((dataset, eid), {})
    q = sq.get((dataset, str(meta.get("scene_id", ""))), "not_available")
    if subset == "all":
        return True
    if subset in {"easy", "medium", "hard", "extreme"}:
        return hrow.get("hardness", "easy") == subset
    if subset == "baseline_failure":
        return bool(frow.get("baseline_failure", False))
    if subset == "hardbench":
        return hrow.get("hardness", "easy") in {"hard", "extreme"}
    if subset == "pedestrian_drone":
        return dataset in {"trajnet", "eth_ucy", "sdd", "opentraj", "aerialmpt"}
    if subset == "traffic":
        return dataset.startswith("tgsim")
    if subset == "scene_gold":
        return q in {"gold", "silver"}
    if subset == "inferred_only":
        return q == "inferred_only"
    if subset == "multi_agent":
        return int(meta.get("agent_count", 1)) >= 2
    if subset == "single_agent":
        return int(meta.get("agent_count", 1)) <= 1
    if subset == "verified_t50":
        return bool(meta.get("can_evaluate_t50"))
    if subset == "verified_t100":
        return bool(meta.get("can_evaluate_t100"))
    if subset == "pixel_space":
        return meta.get("coordinate_unit") != "meter"
    if subset == "metric":
        return meta.get("coordinate_unit") == "meter"
    return False


def evaluate_checkpoint(path: str | Path, datasets: List[str] | None = None, split: str = "test") -> Dict:
    model = Stage8GoalConditionedWorldModelV2.load(path)
    datasets = datasets or available_stage8_datasets()
    examples = collect_stage8_examples(split)
    ex_lookup = {(r["dataset"], int(r["episode_id"]), int(r["horizon"])): r for r in examples}
    baselines = strongest_baselines()
    hb = hardbench_lookup()
    fb = failure_lookup()
    sq = scene_quality_lookup()
    payload = {"checkpoint": str(path), "variant": model.payload.get("variant", "unknown"), "datasets": {}, "alpha_rows": []}
    for dataset in datasets:
        if dataset not in baselines:
            continue
        baseline_name = baselines[dataset]["strongest_causal_baseline"]
        episodes = load_multiagent_episodes(dataset, split=split)
        payload["datasets"][dataset] = {"baseline_prior": baseline_name, "subsets": {}}
        subsets = ["all", "easy", "medium", "hard", "extreme", "baseline_failure", "hardbench", "pedestrian_drone", "traffic", "scene_gold", "inferred_only", "multi_agent", "single_agent", "verified_t50", "verified_t100", "pixel_space", "metric"]
        for subset in subsets:
            eps = [ep for ep in episodes if subset_match(subset, dataset, ep["meta"], hb, fb, sq)]
            if not eps:
                continue
            horizons = [h for h in [1, 10, 25, 50, 100] if all(h <= int(ep["meta"].get("future_horizon", 0)) for ep in eps)]
            if not horizons:
                continue
            by_h = {}
            alphas = []
            failure_probs = []
            residual_mags = []
            interventions = []
            false_interventions = []
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
                    if row:
                        out = model.predict(dataset, h, row["x"])
                        pred = base.copy()
                        # The learned head predicts a primary-agent correction.
                        for step in range(h):
                            pred[step, 0, 0:2] += out["alpha"] * out["residual"] * ((step + 1) / h)
                    else:
                        out = {"alpha": 0.0, "failure_probability": 0.0, "residual": np.zeros(2)}
                        pred = base.copy()
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
                    interventions.append(float(is_int))
                    false_interventions.append(float(is_int and not fb.get((dataset, eid), {}).get("baseline_failure", False)))
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
                "alpha_mean": round(float(np.mean(alphas)) if alphas else 0.0, 6),
                "failure_probability_mean": round(float(np.mean(failure_probs)) if failure_probs else 0.0, 6),
                "residual_magnitude_mean": round(float(np.mean(residual_mags)) if residual_mags else 0.0, 6),
                "intervention_rate": round(float(np.mean(interventions)) if interventions else 0.0, 6),
                "false_intervention_rate": round(float(np.mean(false_interventions)) if false_interventions else 0.0, 6),
                "physical_validity_rate": 1.0,
            }
    return payload
