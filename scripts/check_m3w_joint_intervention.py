"""Synthetic mechanism checks and real-data coordinate roundtrips, not accuracy evaluation."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_unification.m3w_causal_recordings import RecordingWindows, causal_baselines, restore_scene_rollouts
from src.world_model.m3w_joint_intervention import (
    InterventionProblem, past_proximity_edges, proximity_cost_table, select_interventions, screen_cluster_risks,
)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def synthetic_controls() -> dict:
    floor = np.array([[[0., 0.]], [[2., 0.]], [[10., 0.]]])
    candidate = np.array([[[2., 0.]], [[4., 0.]], [[20., 0.]]])
    edges = past_proximity_edges(floor[:, 0], radius=3.)
    cost = proximity_cost_table(floor, candidate, edges, distance_threshold=1.)
    p = InterventionProblem(np.array([.8, -.1, -2.]), np.zeros(3), np.ones(3, dtype=bool),
                            edges, cost, 2., .1, 3)
    results = {mode: select_interventions(p, mode=mode) for mode in
               ("floor", "uncontrolled", "independent", "scene_uniform", "joint")}
    # This separate comparison really matches cardinality; a cap alone would not.
    matched = replace(p, expected_gain=np.array([.8, -.1, .5]), pair_weight=.3, max_interventions=2)
    matched_results = {mode: select_interventions(matched, mode=mode) for mode in ("independent", "joint")}
    assert results["joint"]["switch"].tolist() == [True, True, False]
    assert all(result["switch"].sum() == 2 for result in matched_results.values())
    assert matched_results["joint"]["mean_pair_proxy"] < matched_results["independent"]["mean_pair_proxy"]
    return {"scope": "hand_constructed_scores_no_learning_no_real_world_lift",
            "control_arms": results, "matched_cardinality_example": matched_results}


def exhaustive_checks() -> dict:
    rng = np.random.default_rng(20260916)
    differences, timings = [], []
    for _ in range(80):
        n = 7
        edges = np.array([(i, j) for i in range(n) for j in range(i+1, n)], dtype=int)
        cost = rng.uniform(0, 1, (len(edges), 2, 2))
        cost[:, 0, 0] = 0
        p = InterventionProblem(rng.normal(size=n), rng.uniform(0, .4, n), rng.random(n) > .1,
                                edges, cost, .8, .12, 4)
        start = time.perf_counter()
        selected = select_interventions(p, mode="joint")
        timings.append(time.perf_counter()-start)
        objectives = []
        for bits in product((0, 1), repeat=n):
            x = np.array(bits)
            if x.sum() > 4 or np.any(x > p.supported) or (x*np.maximum(p.expected_harm, -p.expected_gain)).mean() > .12:
                continue
            pair = cost[np.arange(len(edges)), x[edges[:, 0]], x[edges[:, 1]]].mean()
            objectives.append(-(x*p.expected_gain).mean()+.8*pair)
        differences.append(abs(selected["objective"]-min(objectives)))
    assert max(differences) < 1e-8
    return {"scenes": len(differences), "agents_each": 7, "max_objective_error": max(differences),
            "median_seconds": float(np.median(timings)), "max_seconds": max(timings)}


def real_coordinate_checks() -> dict:
    cache = ROOT / "data/stage_cvpr2027_causal"
    manifest = json.loads((cache / "manifest.json").read_text())
    cases, agents, maximum = 0, 0, 0.
    for record in manifest["recordings"]:
        ds = RecordingWindows(cache / record["id"])
        for idx in np.linspace(0, len(ds)-1, 4, dtype=int):
            identity = ds.identity(int(idx))
            scene = ds.get_scene_inputs(identity["frame_id"], identity["horizon_raw"])
            if not scene["agents"]:
                continue
            cases += 1
            agents += len(scene["agents"])
            for k in range(len(record["schema"]["baseline_names"])):
                predictions = {a["agent_id"]: a["inputs"]["baseline_rollouts"][k] for a in scene["agents"]}
                common = restore_scene_rollouts(scene, predictions)
                for a, predicted in zip(scene["agents"], common["xy_dataset_local"]):
                    past = ds.points[(ds.points[:, 1] == a["agent_id"]) & (ds.points[:, 0] <= scene["frame_id"])][-8:]
                    expected = causal_baselines(past[:, 2:4], past[:, 0], common["frame_offsets"])[k]
                    error = float(np.abs(expected-predicted).max())
                    maximum = max(maximum, error)
                    np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-5)
    return {"scene_queries": cases, "agent_queries": agents, "baselines_each": 7,
            "maximum_absolute_roundtrip_error_dataset_local": maximum,
            "future_labels_read": False, "accuracy_evaluation": "not_run"}


def calibration_support_check() -> list[dict]:
    rows = []
    for n in (6, 20, 100, 1000, 10000):
        result = screen_cluster_risks(losses=np.zeros((n, 64, 2)),
                                     cluster_ids=[f"synthetic_{i}" for i in range(n)],
                                     fitted_cluster_ids=["synthetic_train"], policy_ids=[f"policy_{i}" for i in range(64)],
                                     lower=np.zeros(2), upper=np.ones(2), tolerance=np.full(2, .02), delta=.05)
        rows.append({"synthetic_independent_clusters": n, "accepted_count": int(result["accepted"].sum()),
                     "upper_bound_zero_empirical_loss": float(result["upper_risk_bound"][0, 0])})
    return rows


def main() -> None:
    start = time.perf_counter()
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "result_source": "fresh_run",
               "scope": "solver_geometry_and_statistical_assumptions_engineering_checks",
               "synthetic": synthetic_controls(), "exhaustive": exhaustive_checks(),
               "real_coordinate_roundtrip": real_coordinate_checks(),
               "illustrative_calibration_support": calibration_support_check(),
               "calibration_example_is_not_the_easy_degradation_metric": True,
               "formal_protocol_selected": False, "learned_predictor_trained": False,
               "real_world_improvement": "not_run", "submission_ready": False,
               "stage5c_executed": False, "smc_enabled": False}
    payload["source_hashes"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                                ["src/world_model/m3w_joint_intervention.py", "src/data_unification/m3w_causal_recordings.py",
                                 "scripts/check_m3w_joint_intervention.py"]}
    payload["elapsed_seconds"] = time.perf_counter()-start
    directory = ROOT / "outputs/publication_readiness_2026_09/joint_intervention"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mechanism_checks.json").write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False)+"\n")
    print(json.dumps(jsonable({k: payload[k] for k in ("exhaustive", "real_coordinate_roundtrip", "illustrative_calibration_support", "elapsed_seconds")}), indent=2))


if __name__ == "__main__":
    main()
