from __future__ import annotations

from src import stage43_t100_validation_shift_forensics as cn


def test_compare_val_test_flags_allowed_family_that_fails_current_test() -> None:
    val = {
        "UCY": {
            "rows": 100,
            "candidate_t100_full_waypoint_ade_improvement_vs_floor": 0.02,
            "easy_degradation_vs_floor": 0.0,
            "easy_ratio": 0.2,
        }
    }
    test = {
        "UCY": {
            "rows": 200,
            "candidate_t100_full_waypoint_ade_improvement_vs_floor": -0.04,
            "easy_degradation_vs_floor": 0.21,
            "easy_ratio": 0.5,
        }
    }
    result = cn._compare_val_test(val, test, ["UCY"], max_easy_degradation=0.02)
    assert result["UCY"]["allowed_family_failed_current_test"] is True
    assert result["UCY"]["reason"] == "validation_allowed_but_test_negative_or_easy_harm"
    assert result["UCY"]["lift_drop"] < 0.0
    assert result["UCY"]["easy_degradation_increase"] > 0.0


def test_set_overlap_reports_jaccard_and_examples() -> None:
    result = cn._set_overlap(["a", "b", "c"], ["b", "c", "d"])
    assert result["intersection_count"] == 2
    assert abs(result["jaccard"] - 0.5) < 1e-9
    assert result["left_only_examples"] == ["a"]
    assert result["right_only_examples"] == ["d"]


def test_root_causes_include_failed_allowed_family_and_low_overlap() -> None:
    comparison = {
        "UCY": {
            "allowed_family_failed_current_test": True,
            "test_lift": -0.01,
            "test_easy_degradation": 0.21,
            "easy_ratio_shift": 0.3,
        }
    }
    support = {
        "val_test_overlap": {
            "source_file_jaccard": 0.1,
            "scene_jaccard": 0.2,
        }
    }
    causes = cn._root_causes(comparison, support)
    assert "validation_allowed_family_failed_current_test" in causes
    assert "UCY_test_lift_nonpositive" in causes
    assert "UCY_test_easy_harm" in causes
    assert "low_val_test_source_file_overlap" in causes
    assert "low_val_test_scene_overlap" in causes


def test_gate_passes_for_ucy_shift_blocker_payload() -> None:
    payload = {
        "source": cn.SOURCE,
        "input_verdicts": {
            "stage43_cm": "stage43_cm_current_matrix_t100_source_family_gate_pass_keep_floor",
        },
        "selected_model_replayed": {
            "model_hash_matches_stage43_cm": True,
        },
        "support": {
            "val_h100_rows": 100,
            "test_h100_rows": 200,
        },
        "validation_test_shift_by_source_family": {
            "UCY": {
                "allowed_family_failed_current_test": True,
            }
        },
        "root_causes": ["validation_allowed_family_failed_current_test"],
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
    gate = cn._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_cn_t100_validation_shift_forensics_pass_ucy_shift_blocker"
    assert gate["deploy_t100"] is False
