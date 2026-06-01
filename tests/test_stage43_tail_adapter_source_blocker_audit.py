from __future__ import annotations

from src import stage43_tail_adapter_source_blocker_audit as ba


def test_classify_validation_blockers_maps_allowed_and_blocked() -> None:
    table = {
        "UCY|50": {
            "rows": 10,
            "allowed": True,
            "full_waypoint_ade_improvement_vs_floor": 0.2,
            "easy_degradation_vs_floor": 0.0,
            "reason": "allowed_positive",
        },
        "UCY|100": {
            "rows": 5,
            "allowed": False,
            "full_waypoint_ade_improvement_vs_floor": -0.1,
            "easy_degradation_vs_floor": 0.0,
            "reason": "blocked_validation_nonpositive",
        },
    }
    by_family = ba._classify_validation_blockers(table)
    assert by_family["UCY"]["validation_rows"] == 15
    assert [row["horizon"] for row in by_family["UCY"]["allowed_horizons"]] == [50]
    assert [row["horizon"] for row in by_family["UCY"]["blocked_horizons"]] == [100]
    assert by_family["UCY"]["h100_blocked"] is True
    assert by_family["UCY"]["block_reasons"] == {"blocked_validation_nonpositive": 1}


def test_test_source_rows_separates_positive_and_floor_blocked() -> None:
    by_source_summary = {
        "worst_sources": [
            {
                "slice": "/tmp/TrajNet/Train/biwi/biwi_hotel.txt",
                "rows": 100,
                "full_waypoint_ade_improvement_vs_floor": 0.0,
                "endpoint_fde_improvement_vs_floor": 0.0,
                "ungated_full_waypoint_ade_improvement_vs_floor": -5.0,
                "easy_degradation_vs_floor": 0.0,
                "switch_rate": 0.0,
            }
        ],
        "best_sources": [
            {
                "slice": "/tmp/UCY/zara01.txt",
                "rows": 200,
                "full_waypoint_ade_improvement_vs_floor": 0.5,
                "endpoint_fde_improvement_vs_floor": 0.4,
                "ungated_full_waypoint_ade_improvement_vs_floor": 0.45,
                "easy_degradation_vs_floor": 0.0,
                "switch_rate": 0.8,
            }
        ],
    }
    rows = ba._test_source_rows(by_source_summary)
    assert {row["status"] for row in rows} == {"safe_floor_blocked", "positive_switched"}
    blocked = next(row for row in rows if row["status"] == "safe_floor_blocked")
    assert blocked["ungated_improvement"] == -5.0


def test_blocked_source_diagnosis_marks_catastrophic_ungated_transfer() -> None:
    source_rows = [
        {
            "source_file": "/tmp/TrajNet/Train/mot/PETS09-S2L1.txt",
            "family": "TrajNet_mot",
            "rows": 10,
            "selected_improvement": 0.0,
            "endpoint_improvement": 0.0,
            "ungated_improvement": -2.0,
            "easy_degradation": 0.0,
            "switch_rate": 0.0,
            "status": "safe_floor_blocked",
        }
    ]
    validation = {
        "TrajNet_mot": {
            "allowed_horizons": [],
            "blocked_horizons": [{"horizon": 50}],
            "block_reasons": {"blocked_validation_nonpositive": 1},
            "h100_blocked": False,
            "validation_rows": 10,
        }
    }
    blocked = ba._blocked_source_diagnosis(source_rows, validation)
    assert blocked[0]["diagnosis"] == "floor_required_ungated_catastrophic_negative_transfer"


def test_gate_passes_for_floor_necessity_and_no_overclaim() -> None:
    payload = {
        "input_verdicts": {
            "stage43_p": "stage43_p_tail_horizon_adapter_pass_t100_still_fallback",
            "stage43_az": "stage43_az_tail_adapter_reviewer_replay_pass",
        },
        "blocked_sources": [{"diagnosis": "floor_required_ungated_catastrophic_negative_transfer"}],
        "validation_by_family": {"TrajNet_biwi": {"blocked_horizons": [50]}},
        "next_required_actions": ["a", "b", "c"],
        "summary": {
            "test_source_count": 4,
            "positive_switched_source_count": 2,
            "safe_floor_blocked_source_count": 2,
            "floor_necessity_supported_for_blocked_sources": True,
            "uniform_positive_transfer_claim_allowed": False,
            "uniform_nonnegative_transfer_supported": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "uniform_positive_external_transfer_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = ba._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_ba_tail_adapter_source_blocker_audit_pass"
