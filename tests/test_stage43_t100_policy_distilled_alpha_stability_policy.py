from __future__ import annotations

from src import stage43_t100_policy_distilled_alpha_stability_policy as de


def _seed(original_t100: float, bounded_t100: float, original_min: float, bounded_min: float) -> dict:
    return {
        "max_replay_diff": 0.0,
        "original_test_metrics": {
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": original_t100,
        },
        "original_test_group_summary": {
            "min_without_any_group_t100": original_min,
        },
        "selected_variant": {
            "variant": "alpha_cap_0_75",
            "eligible_candidate_count": 3,
            "selected_policy": {"alpha": 0.75},
            "test_metrics": {
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": bounded_t100,
                "easy_degradation_vs_floor": 0.0,
                "switch_rate": 0.1,
            },
            "test_group_summary": {
                "min_without_any_group_t100": bounded_min,
            },
        },
        "test_delta_vs_original": {
            "t100": bounded_t100 - original_t100,
            "min_without_group_t100": bounded_min - original_min,
        },
    }


def test_aggregate_detects_alpha_stability_group_repair() -> None:
    runs = [_seed(0.002, 0.0018, -0.001, 0.001) for _ in range(3)]
    agg = de._aggregate(
        runs,
        {"aggregate": {"t100": {"mean": 0.002}, "min_without_group_t100": {"mean": -0.001}}},
        {"aggregate": {"robust_t100": {"mean": 0.0015}, "robust_min_without_group_t100": {"mean": 0.0004}}},
        {"aggregate": {"guarded_t100": {"mean": 0.0021}, "guarded_min_without_group_t100": {"mean": -0.0002}, "all_guarded_min_without_group_positive": False}},
    )
    assert agg["all_replay_exact"] is True
    assert agg["all_bounded_min_without_group_positive"] is True
    assert agg["repairs_dd_seed_fragility"] is True
    assert agg["beats_cz_t100_mean"] is True


def test_gate_passes_when_bounded_policy_repairs_dd_seed_fragility() -> None:
    runs = [_seed(0.002, 0.0018, -0.001, 0.001) for _ in range(3)]
    agg = de._aggregate(
        runs,
        {"aggregate": {"t100": {"mean": 0.002}, "min_without_group_t100": {"mean": -0.001}}},
        {"aggregate": {"robust_t100": {"mean": 0.0015}, "robust_min_without_group_t100": {"mean": 0.0004}}},
        {"aggregate": {"guarded_t100": {"mean": 0.0021}, "guarded_min_without_group_t100": {"mean": -0.0002}, "all_guarded_min_without_group_positive": False}},
    )
    payload = {
        "stage43_dd_precondition": {"verdict": "stage43_dd_t100_policy_distilled_group_guard_mean_improves_dc_seed_fragile"},
        "result_source": "fresh_bounded_alpha_policy_selection_on_dc_head",
        "seed_runs": runs,
        "selection_protocol": {"test_threshold_tuning": False, "max_alpha_cap": 0.75},
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
    gate = de._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_de_t100_alpha_stability_policy_repairs_group_fragility_diagnostic"
