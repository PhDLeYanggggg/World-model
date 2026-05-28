import numpy as np

from src import stage42_context_slice_policy_conservative_repair as jb


def test_candidate_supported_requires_core_preservation():
    baseline = {
        "all_improvement": 0.10,
        "t50_improvement": 0.10,
        "t100_raw_frame_diagnostic_improvement": 0.10,
        "hard_failure_improvement": 0.10,
    }
    good = {
        "all_improvement": 0.112,
        "t50_improvement": 0.10,
        "t100_raw_frame_diagnostic_improvement": 0.10,
        "hard_failure_improvement": 0.10,
        "easy_degradation": 0.0,
    }
    drops_core = dict(good, t50_improvement=0.08)
    assert jb._candidate_supported(good, baseline, jb.MIN_VAL_SLICE_ROWS)
    assert not jb._candidate_supported(drops_core, baseline, jb.MIN_VAL_SLICE_ROWS)
    assert not jb._candidate_supported(good, baseline, jb.MIN_VAL_SLICE_ROWS - 1)


def test_greedy_select_rules_accepts_incremental_safe_rule():
    n = 6
    data = {
        "dataset": np.asarray(["A"] * n),
        "horizon": np.asarray([50] * n),
        "hard": np.asarray([True, True, False, False, False, False]),
        "failure": np.asarray([False] * n),
        "easy": np.asarray([False, False, True, True, True, True]),
    }
    shared = {"split": np.asarray(["val"] * n), "data": data}
    outputs = {
        "tree_baseline_family_residual": {
            "selected_ade": np.asarray([9.0] * n),
            "selected_fde": np.asarray([9.0] * n),
            "switch": np.asarray([False] * n),
            "floor_ade": np.asarray([10.0] * n),
            "floor_fde": np.asarray([10.0] * n),
        },
        "tree_full_residual": {
            "selected_ade": np.asarray([7.0] * n),
            "selected_fde": np.asarray([7.0] * n),
            "switch": np.asarray([True] * n),
            "floor_ade": np.asarray([10.0] * n),
            "floor_fde": np.asarray([10.0] * n),
        },
    }
    slices = {"horizon:50": np.asarray([True] * n)}
    candidates = [
        {
            "slice": "horizon:50",
            "trial": "tree_full_residual",
            "supported_on_val": True,
            "candidate_score": 1.0,
            "val_rows": n,
        }
    ]
    old_min = jb.MIN_VAL_SLICE_ROWS
    try:
        jb.MIN_VAL_SLICE_ROWS = 3
        selected, diagnostics = jb._greedy_select_rules(shared, outputs, slices, candidates)
    finally:
        jb.MIN_VAL_SLICE_ROWS = old_min
    assert diagnostics["selected_rule_count"] == 1
    assert selected[0]["slice"] == "horizon:50"


def test_gate_records_negative_when_context_regresses_core_metric():
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "selection_diagnostics": {
            "selection_source": "validation_only_greedy_incremental",
            "inference_safe_slice_filter": True,
            "selected_rule_count": 1,
        },
        "selected_rules": [{"slice": "horizon:50"}],
        "summary": {
            "metrics": {
                "conservative_context_policy": {"rows": 47458, "easy_degradation": 0.0},
                "delta_vs_baseline_family": {
                    "all_improvement": 0.01,
                    "t50_improvement": -0.01,
                    "t100_raw_frame_diagnostic_improvement": 0.0,
                    "hard_failure_improvement": 0.0,
                },
            }
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_slice_thresholds": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = jb._gate(result)
    assert not gate["gates"]["no_core_metric_regression_vs_baseline_family"]
    assert gate["verdict"] == "stage42_jb_conservative_context_policy_not_promotable"


def test_stage42_jb_run_completes_or_uses_cache():
    result = jb.run_stage42_context_slice_policy_conservative_repair(use_cached=True)
    if not result:
        result = jb.run_stage42_context_slice_policy_conservative_repair()
    assert result["stage42_jb_gate"]["gates"]["validation_greedy_selection"]
    assert result["stage42_jb_gate"]["gates"]["test_once_policy_evaluated"]
    assert result["no_leakage"]["future_waypoint_input"] is False
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
