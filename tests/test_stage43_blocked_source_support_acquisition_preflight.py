from __future__ import annotations

from src import stage43_blocked_source_support_acquisition_preflight as be


def test_local_candidate_family_maps_mot_like_sources() -> None:
    assert be._local_candidate_family("PETS-2009-S2L1") == "mot_like"
    assert be._local_candidate_family("Town-Center") == "mot_like_or_external_topdown_support"
    assert be._local_candidate_family("Wild-Track") == "mot_like_or_external_topdown_support"


def test_local_candidate_record_keeps_terms_blocker() -> None:
    row = {
        "dataset_name": "PETS-2009-S2L1",
        "root": "/tmp/PETS",
        "parseable": True,
        "stats": {"point_rows": 100, "agent_tracks": 5, "t50_rows": 10, "t100_rows": 0},
        "calibration_file_count": 3,
        "coordinate_unit": "image_pixel_bbox_bottom_center",
        "metric_status": "calibration_present_unintegrated",
        "legal_auto_convert_allowed": False,
        "conversion_status": "not_converted_license_or_projection_guard",
    }
    out = be._local_candidate_record(row)
    assert out["support_family"] == "mot_like"
    assert out["technical_support_candidate"] is True
    assert out["conversion_ready_now"] is False
    assert "terms_or_license_not_confirmed_for_benchmark_conversion" in out["blockers"]
    assert "not_converted_into_stage43_feature_store" in out["blockers"]


def test_family_readiness_keeps_repair_training_disallowed() -> None:
    blocked = [
        {
            "family": "TrajNet_biwi",
            "support_candidate_exists_in_raw_scan": True,
        },
        {
            "family": "TrajNet_mot",
            "support_candidate_exists_in_raw_scan": False,
        },
    ]
    local = [
        {
            "dataset_name": "PETS-2009-S2L1",
            "support_family": "mot_like",
            "technical_support_candidate": True,
            "conversion_ready_now": False,
        }
    ]
    readiness = be._family_readiness(blocked, local)
    assert readiness["TrajNet_biwi"]["repair_training_allowed_now"] is False
    assert readiness["TrajNet_mot"]["repair_training_allowed_now"] is False
    assert readiness["TrajNet_mot"]["technical_candidate_count"] == 1
    assert readiness["overall"]["conversion_ready_now_count"] == 0


def test_gate_passes_for_acquisition_preflight_without_conversion_or_training() -> None:
    payload = {
        "input_verdicts": {
            "stage43_bc": "stage43_bc_blocked_family_support_scan_pass",
            "stage43_bd": "stage43_bd_biwi_support_rebuild_preflight_pass",
        },
        "summary": {
            "blocked_family_count": 2,
            "local_candidate_count": 3,
            "local_technical_support_candidate_count": 3,
            "local_conversion_ready_now_count": 0,
            "repair_training_allowed_now_count": 0,
        },
        "family_readiness": {
            "TrajNet_biwi": {
                "repair_training_allowed_now": False,
                "reason": "requires independent source support",
            },
            "TrajNet_mot": {
                "repair_training_allowed_now": False,
                "technical_candidate_count": 2,
                "conversion_ready_count": 0,
            },
        },
        "next_required_actions": ["a", "b", "c", "d"],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "conversion_executed": False,
            "training_executed": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "external_not_run_written_as_success": False,
            "blocked_source_repair_success_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = be._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_be_blocked_source_support_acquisition_preflight_pass"
