from __future__ import annotations

from src import stage43_t100_policy_distilled_group_stability_guard as dd


def _seed(base_t100: float, guarded_t100: float, base_min: float, guarded_min: float) -> dict:
    return {
        "max_replay_diff": 0.0,
        "validation_variants": [{"variant": "floor"}, {"variant": "source_val_positive"}, {"variant": "scene_val_positive"}, {"variant": "domain_val_positive"}],
        "base_test_metrics": {
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": base_t100,
        },
        "guarded_test_metrics": {
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": guarded_t100,
            "easy_degradation_vs_floor": 0.0,
            "switch_rate": 0.1,
        },
        "base_test_group_summary": {
            "min_without_any_group_t100": base_min,
        },
        "guarded_test_group_summary": {
            "min_without_any_group_t100": guarded_min,
        },
        "test_delta_vs_base": {
            "t100": guarded_t100 - base_t100,
            "min_without_group_t100": guarded_min - base_min,
        },
        "selected_variant": {"variant": "source_val_positive"},
    }


def test_aggregate_detects_group_fragility_reduction() -> None:
    runs = [_seed(0.002, 0.0021, -0.001, 0.001) for _ in range(3)]
    agg = dd._aggregate(
        runs,
        {"aggregate": {"t100": {"mean": 0.002}, "min_without_group_t100": {"mean": -0.001}}},
        {"aggregate": {"robust_t100": {"mean": 0.0018}, "robust_min_without_group_t100": {"mean": 0.0005}}},
    )
    assert agg["all_replay_exact"] is True
    assert agg["group_fragility_reduced"] is True
    assert agg["beats_dc_t100_mean"] is True
    assert agg["beats_cz_min_without_group_mean"] is True


def test_gate_reports_improves_dc_when_t100_and_group_both_improve() -> None:
    runs = [_seed(0.002, 0.0021, -0.001, 0.001) for _ in range(3)]
    agg = dd._aggregate(
        runs,
        {"aggregate": {"t100": {"mean": 0.002}, "min_without_group_t100": {"mean": -0.001}}},
        {"aggregate": {"robust_t100": {"mean": 0.0018}, "robust_min_without_group_t100": {"mean": 0.0005}}},
    )
    payload = {
        "stage43_dc_precondition": {"verdict": "stage43_dc_t100_policy_distilled_head_beats_cz_diagnostic"},
        "result_source": "fresh_validation_group_support_guard_on_policy_distilled_head",
        "seed_runs": runs,
        "selection_protocol": {"test_threshold_tuning": False},
        "aggregate": agg,
        "deploy_on_current_heldout": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    gate = dd._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_dd_t100_policy_distilled_group_guard_improves_dc"
