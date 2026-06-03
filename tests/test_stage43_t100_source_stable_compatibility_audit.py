from __future__ import annotations

from src import stage43_t100_source_stable_compatibility_audit as cl


def test_source_level_compatibility_blocks_global_integration_for_small_split() -> None:
    stage_t = {
        "stage43_t_gate": {
            "positive_h100_dynamics_signal": True,
            "deploy_source_stable_h100_specialist": True,
        },
        "source_level_split": {
            "test_rows": 100,
            "test_sources": ["OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt"],
        },
        "training_protocol": {"family": "TrajNet_crowds"},
        "source_stable_h100_test_metrics": {
            "full_waypoint_ade_improvement_vs_floor": 0.03,
            "endpoint_fde_improvement_vs_floor": -0.01,
            "easy_degradation_vs_floor": 0.0,
        },
    }
    stage_ck = {
        "stage43_ck_gate": {
            "t100_positive_success": False,
            "deploy_t100_causal_specialist": False,
        },
        "test_metrics_with_causal_specialist": {
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        },
    }
    matrix = {"test_rows": 10000}
    result = cl._source_level_compatibility(stage_t, stage_ck, matrix)
    assert result["stage43_t_positive_local_signal"] is True
    assert result["same_split_scope_as_current_matrix"] is False
    assert result["can_integrate_as_global_t100_deployment"] is False
    assert result["compatibility_reason"] == "local_source_level_positive_signal_not_current_full_matrix_deployment"


def test_feature_contract_reports_protocol_clean_but_feature_names_missing() -> None:
    stage_t = {
        "training_protocol": {
            "future_waypoints_as_labels_only": True,
            "test_threshold_tuning": False,
            "selection_data": "source_level_validation_only",
            "family": "TrajNet_crowds",
        }
    }
    result = cl._feature_contract(stage_t)
    assert result["feature_names_available_in_stage43_t_report"] is False
    assert result["future_waypoints_label_only"] is True
    assert result["reported_test_threshold_tuning"] is False
    assert result["denied_protocol_fragments"] == []


def test_gate_passes_for_local_positive_but_global_floor() -> None:
    payload = {
        "input_verdicts": {
            "stage43_s": "stage43_s_t100_source_coverage_preflight_pass",
        },
        "compatibility": {
            "stage43_t_positive_local_signal": True,
            "stage43_t_easy_degradation": 0.0,
            "stage43_ck_global_t100_success": False,
            "stage43_ck_deploy_t100": False,
            "same_split_scope_as_current_matrix": False,
            "stage43_t_row_ratio_vs_current_matrix": 0.02,
        },
        "feature_contract": {
            "future_waypoints_label_only": True,
            "reported_test_threshold_tuning": False,
            "denied_protocol_fragments": [],
        },
        "claim_decision": {
            "global_t100_deployment_allowed": False,
            "uniform_t100_success_allowed": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = cl._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_cl_t100_source_stable_compatibility_pass_local_only"
    assert gate["local_t100_positive_signal_allowed"] is True
    assert gate["global_t100_deployment_allowed"] is False
