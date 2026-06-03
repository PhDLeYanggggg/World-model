from __future__ import annotations

import numpy as np

from src import stage43_t100_source_scene_support_gate as co


class _TinySplit:
    def __init__(self) -> None:
        self.horizon = np.asarray([100, 100, 50, 100], dtype=np.int64)
        self.floor_ade = np.asarray([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        self.floor_fde = np.asarray([11.0, 11.0, 11.0, 11.0], dtype=np.float32)
        self.easy = np.asarray([False, False, False, True], dtype=bool)
        self.hard = np.asarray([True, False, False, False], dtype=bool)
        self.failure = np.asarray([False, False, False, False], dtype=bool)
        self.source_file = np.asarray(["a", "a", "a", "b"], dtype=str)
        self.scene_id = np.asarray(["s1", "s1", "s1", "s2"], dtype=str)


def test_validation_support_rules_allow_only_positive_easy_safe_supported_slice() -> None:
    ds = _TinySplit()
    candidate = np.asarray([8.0, 8.0, 8.0, 12.0], dtype=np.float32)
    table, allowed = co._validation_support_rules(
        ds,  # type: ignore[arg-type]
        candidate,
        key_values=ds.source_file,
        min_support_rows=2,
        min_improvement=0.0,
        max_easy_degradation=0.02,
    )
    assert "a" in allowed
    assert "b" not in allowed
    assert table["a"]["reason"] == "allowed_by_validation"
    assert table["b"]["reason"] == "blocked_insufficient_validation_support"


def test_apply_source_scene_support_blocks_unsupported_t100_rows() -> None:
    ds = _TinySplit()
    candidate_ade = np.asarray([8.0, 8.0, 8.0, 8.0], dtype=np.float32)
    candidate_fde = np.asarray([9.0, 9.0, 9.0, 9.0], dtype=np.float32)
    selected_ade, selected_fde, switch, support = co._apply_source_scene_support(
        ds,  # type: ignore[arg-type]
        candidate_ade,
        candidate_fde,
        allowed_source_files={"a"},
        allowed_scenes=set(),
    )
    assert switch.tolist() == [True, True, False, False]
    assert float(selected_ade[0]) == 8.0
    assert float(selected_fde[3]) == 11.0
    assert support["h100_rows"] == 3
    assert support["switched_h100_rows"] == 2
    assert support["blocked_h100_rows"] == 1


def test_support_overlap_zero_when_validation_and_test_sources_disjoint() -> None:
    val = _TinySplit()
    test = _TinySplit()
    test.source_file = np.asarray(["x", "x", "x", "y"], dtype=str)
    result = co._support_overlap(val, test, key="source_file")  # type: ignore[arg-type]
    assert result["intersection_count"] == 0
    assert result["jaccard"] == 0.0
    assert result["test_only_count"] == 2


def test_gate_passes_when_unsafe_family_rule_is_blocked_by_support_floor() -> None:
    payload = {
        "source": co.SOURCE,
        "input_verdicts": {
            "stage43_cm": "stage43_cm_current_matrix_t100_source_family_gate_pass_keep_floor",
            "stage43_cn": "stage43_cn_t100_validation_shift_forensics_pass_ucy_shift_blocker",
        },
        "selected_model_replayed": {"model_hash_matches_stage43_cm": True},
        "validation_source_file_rules": {"table": {"a": {}}},
        "validation_scene_rules": {"table": {"s1": {}}},
        "support_overlap": {
            "source_file": {"jaccard": 0.0},
            "scene": {"jaccard": 0.0},
        },
        "switch_support": {
            "h100_rows": 10,
            "blocked_h100_rows": 10,
        },
        "raw_family_rule_test_metrics": {
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": -0.03,
        },
        "source_scene_supported_test_metrics": {
            "easy_degradation_vs_floor": 0.0,
            "full_waypoint_ade_improvement_vs_floor": 0.0,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        },
        "deployment_decision": {
            "deploy_source_scene_supported_t100_gate": False,
        },
        "support_rule_protocol": {
            "selection_data": "validation_only",
            "test_threshold_tuning": False,
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
            "t100_deployment_claim": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = co._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_co_t100_source_scene_support_gate_pass_floor_required"
    assert gate["deploy_t100"] is False
