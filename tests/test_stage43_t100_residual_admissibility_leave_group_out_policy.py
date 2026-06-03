from __future__ import annotations

from src import stage43_t100_residual_admissibility_leave_group_out_policy as cz


def test_robust_objective_penalizes_leave_group_failure() -> None:
    metrics = {
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.01,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.02,
        "full_waypoint_ade_improvement_vs_floor": 0.01,
        "easy_degradation_vs_floor": 0.0,
        "switch_rate": 0.1,
    }
    stable = {"min_without_any_group_t100": 0.002, "source_group_flip_count": 0, "scene_group_flip_count": 0, "domain_pair_flip_count": 0}
    fragile = {"min_without_any_group_t100": -0.002, "source_group_flip_count": 0, "scene_group_flip_count": 1, "domain_pair_flip_count": 0}
    assert cz._robust_objective(metrics, stable) > cz._robust_objective(metrics, fragile)


def test_aggregate_detects_group_reduction() -> None:
    runs = [
        {
            "max_replay_diff": 0.0,
            "safe_candidate_count": 2,
            "selected_validation_candidate": {"policy": {"selection_mode": "leave_group_out_robust"}},
            "original_test_metrics": {"t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.001},
            "robust_test_metrics": {
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0009,
                "easy_degradation_vs_floor": 0.0,
                "switch_rate": 0.05,
            },
            "original_test_group_summary": {"min_without_any_group_t100": -0.001},
            "robust_test_group_summary": {"min_without_any_group_t100": 0.0001},
            "test_delta_vs_original": {
                "t100": -0.0001,
                "min_without_group_t100": 0.0011,
                "scene_group_flip_count": -1.0,
            },
        }
        for _ in range(3)
    ]
    agg = cz._aggregate(runs)
    assert agg["all_replay_exact"] is True
    assert agg["group_fragility_reduced"] is True
    assert agg["robust_preserves_easy"] is True


def test_gate_requires_validation_selection() -> None:
    payload = {
        "stage43_cy_precondition": {"verdict": "stage43_cy_t100_group_support_guard_no_repair_keep_diagnostic"},
        "result_source": "fresh_leave_group_out_robust_policy_search",
        "seed_runs": [{}, {}, {}],
        "aggregate": {
            "all_replay_exact": True,
            "safe_candidate_count": {"min": 1},
            "group_fragility_reduced": False,
            "robust_preserves_easy": True,
            "robust_t100_positive_all_seeds": True,
        },
        "selection_protocol": {"test_threshold_tuning": True, "objective": "leave_group_out_min_t100_plus_flip_penalty"},
        "deploy_on_current_heldout": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "long_objective_complete": False,
    }
    gate = cz._gate(payload)
    assert gate["passed"] == gate["total"] - 1
    assert gate["verdict"] == "stage43_cz_t100_leave_group_out_policy_incomplete_keep_diagnostic"
