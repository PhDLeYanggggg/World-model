import numpy as np

from src import stage42_source_level_nonlinear_context_repair as iy


def test_train_ids_are_deterministic_and_capped():
    split = np.asarray(["train"] * 200 + ["val"] * 10 + ["test"] * 10)
    data = {
        "horizon": np.asarray(([10, 25, 50, 100] * 55)[:220]),
        "dataset": np.asarray((["A", "B"] * 110)[:220]),
        "hard": np.asarray(([False, True] * 110)[:220]),
        "failure": np.zeros(220, dtype=bool),
    }
    old_cap = iy.MAX_TREE_TRAIN_ROWS
    try:
        iy.MAX_TREE_TRAIN_ROWS = 40
        a = iy._train_ids(split, data)
        b = iy._train_ids(split, data)
    finally:
        iy.MAX_TREE_TRAIN_ROWS = old_cap
    assert len(a) == 40
    assert np.array_equal(a, b)
    assert np.all(split[a] == "train")


def test_metric_delta_detects_context_lift():
    better = {
        "all_improvement": 0.20,
        "t50_improvement": 0.15,
        "t100_raw_frame_diagnostic_improvement": 0.02,
        "hard_failure_improvement": 0.18,
        "easy_degradation": 0.0,
        "switch_rate": 0.3,
        "harm_over_fallback": -0.2,
    }
    base = {
        "all_improvement": 0.18,
        "t50_improvement": 0.12,
        "t100_raw_frame_diagnostic_improvement": 0.02,
        "hard_failure_improvement": 0.16,
        "easy_degradation": 0.0,
        "switch_rate": 0.2,
        "harm_over_fallback": -0.1,
    }
    delta = iy._metric_delta(better, base)
    assert delta["all_improvement"] > iy.MIN_CONTEXT_LIFT
    assert delta["t50_improvement"] > iy.MIN_CONTEXT_LIFT


def test_gate_records_completed_negative_capacity_result():
    metric = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
    }
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "trials": {t["name"]: {"train_rows_used": 100, "trial": t} for t in iy.TREE_TRIALS},
        "summary": {
            "best_trial_metric": metric,
            "capacity_hypothesis_verdict": "stage42_iy_nonlinear_context_capacity_not_sufficient",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = iy._gate(result)
    assert gate["gates"]["tree_trials_complete"]
    assert not gate["gates"]["nonlinear_context_claim_supported"]
    assert gate["verdict"] == "stage42_iy_nonlinear_context_repair_completed_context_not_proven"


def test_stage42_iy_run_completes_or_uses_cache():
    result = iy.run_stage42_source_level_nonlinear_context_repair(use_cached=True)
    if not result:
        result = iy.run_stage42_source_level_nonlinear_context_repair()
    gate = result["stage42_iy_gate"]
    assert gate["gates"]["tree_trials_complete"]
    assert gate["gates"]["context_tree_trial_tested"]
    assert gate["gates"]["full_tree_trial_tested"]
    assert result["summary"]["capacity_hypothesis_verdict"] in {
        "stage42_iy_nonlinear_context_capacity_positive",
        "stage42_iy_nonlinear_context_capacity_not_sufficient",
    }
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
