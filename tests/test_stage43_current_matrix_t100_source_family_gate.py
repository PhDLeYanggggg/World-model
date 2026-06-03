from __future__ import annotations

import numpy as np

from src import stage43_current_matrix_t100_source_family_gate as cm


def test_feature_contract_persists_names_and_blocks_label_like_features() -> None:
    clean = ["history_speed_tail0", "prototype_distance_0", "causal_floor_waypoint_delta_0"]
    result = cm._feature_contract(clean)
    assert result["feature_names_persisted"] is True
    assert result["feature_dim"] == 3
    assert result["denied_feature_name_hits"] == []
    assert result["causal_floor_waypoint_rollout_feature_count"] == 1

    dirty = cm._feature_contract(["history_speed_tail0", "future_endpoint_x", "oracle_margin"])
    assert "future_endpoint_x" in dirty["denied_feature_name_hits"]
    assert "oracle_margin" in dirty["denied_feature_name_hits"]


class _TinySplit:
    def __init__(self) -> None:
        self.source_file = np.asarray(
            [
                "OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara01.txt",
                "OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara01.txt",
                "OpenTraj/datasets/TrajNet/Train/biwi/biwi_eth.txt",
                "OpenTraj/datasets/TrajNet/Train/biwi/biwi_eth.txt",
            ],
            dtype=str,
        )
        self.horizon = np.asarray([100, 100, 100, 100], dtype=np.int64)
        self.floor_ade = np.asarray([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        self.easy = np.asarray([False, False, True, True], dtype=bool)


def test_family_t100_table_allows_only_supported_positive_easy_safe_family() -> None:
    ds = _TinySplit()
    candidate = np.asarray([8.0, 8.0, 12.0, 12.0], dtype=np.float32)
    table, allowed = cm._family_t100_table(
        ds,  # type: ignore[arg-type]
        candidate,
        min_support_rows=2,
        min_improvement=0.0,
        max_easy_degradation=0.02,
    )
    assert "TrajNet_crowds" in allowed
    assert "TrajNet_biwi" not in allowed
    assert table["TrajNet_crowds"]["reason"] == "allowed_by_validation"
    assert table["TrajNet_biwi"]["reason"] == "blocked_validation_nonpositive"


def test_deployment_decision_keeps_floor_when_allowed_family_harms_easy() -> None:
    metrics = {
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.05,
        "easy_degradation_vs_floor": 0.0,
        "full_waypoint_ade_improvement_vs_floor": 0.01,
    }
    table = {
        "TrajNet_crowds": {
            "validation_allowed": True,
            "selected_t100_full_waypoint_ade_improvement_vs_floor": 0.05,
            "easy_degradation_vs_floor": 0.0,
        },
        "TrajNet_biwi": {
            "validation_allowed": True,
            "selected_t100_full_waypoint_ade_improvement_vs_floor": -0.02,
            "easy_degradation_vs_floor": 0.0,
        },
    }
    decision = cm._deployment_decision(metrics, table, {"TrajNet_crowds", "TrajNet_biwi"})
    assert decision["deploy_current_matrix_t100_source_family_gate"] is False
    assert decision["reason"] == "keep_floor_because_current_matrix_test_is_not_uniformly_positive_easy_safe"


def test_gate_passes_keep_floor_for_safe_deployed_floor() -> None:
    payload = {
        "source": cm.SOURCE,
        "input_verdicts": {
            "stage43_cl": "stage43_cl_t100_source_stable_compatibility_pass_local_only",
        },
        "current_matrix_scope": {
            "test_rows": 1000,
            "stage43_cl_local_t100_rows": 100,
            "stage43_at_matrix_test_rows": 1000,
        },
        "feature_contract": {
            "feature_names_persisted": True,
            "feature_dim": 2,
            "feature_names": ["history_speed_tail0", "causal_floor_waypoint_delta_0"],
            "feature_name_hash": "abc",
            "denied_feature_name_hits": [],
            "future_waypoints_label_only": True,
            "baseline_rollout_computed_without_future": True,
        },
        "training_protocol": {
            "selection_data": "validation_only",
            "test_threshold_tuning": False,
        },
        "selected_model": {
            "validation_source_family_t100_table": {"TrajNet_crowds": {"rows": 200}},
        },
        "test_source_family_t100_table": {"TrajNet_crowds": {"rows": 200}},
        "raw_validation_rule_test_metrics": {
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": -0.01,
        },
        "deployed_test_metrics": {
            "easy_degradation_vs_floor": 0.0,
            "full_waypoint_ade_improvement_vs_floor": 0.0,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        },
        "deployment_decision": {
            "deploy_current_matrix_t100_source_family_gate": False,
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
            "global_t100_success_claim": False,
            "uniform_t100_success_claim": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = cm._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_cm_current_matrix_t100_source_family_gate_pass_keep_floor"
    assert gate["deploy_current_matrix_t100_source_family_gate"] is False
