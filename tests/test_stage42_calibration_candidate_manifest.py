from __future__ import annotations

from src.stage42_calibration_candidate_manifest import _gate, classify_candidate


def test_classify_source_specific_candidate_preserves_legal_blocker() -> None:
    row = {
        "dataset_id": "ucy_crowd_original",
        "domain": "UCY",
        "h_matrix_hint_count": 4,
        "time_metadata_hint_count": 1,
        "frame_stride_hint_count": 8,
        "metric_time_subset_hint": True,
        "legal_conversion_ready": False,
    }

    candidate = classify_candidate(row, ["UCY_zara01", "UCY_students03"])

    assert candidate["candidate_class"] == "source_specific_metric_time_candidate_after_terms"
    assert candidate["priority_score"] == 95
    assert candidate["conversion_allowed_now"] is False
    assert "terms/source_identity/path_version_not_confirmed" in candidate["blockers"]
    assert candidate["global_metric_or_seconds_claim_allowed"] is False


def test_classify_trajnet_time_stride_as_dataset_local_only() -> None:
    row = {
        "dataset_id": "trajnetplusplus_official",
        "domain": "TrajNet",
        "h_matrix_hint_count": 0,
        "time_metadata_hint_count": 1,
        "frame_stride_hint_count": 12,
        "legal_conversion_ready": False,
    }

    candidate = classify_candidate(row, [])

    assert candidate["candidate_class"] == "time_stride_candidate_dataset_local_only"
    assert candidate["priority_score"] == 55
    assert "homography_or_coordinate_transform_missing" in candidate["blockers"]


def test_gate_passes_only_with_blocked_global_claims() -> None:
    payload = {
        "stage42_du_verdict": "stage42_du_raw_source_time_geometry_hint_audit_pass",
        "stage42_bn_verdict": "stage42_bn_source_time_geometry_calibration_pass_with_global_claim_blocked",
        "user_action_required_written": True,
        "data_calibration_updated": True,
        "summary": {
            "targets_checked": 7,
            "source_specific_candidate_targets": 2,
            "time_stride_candidate_targets": 1,
            "conversion_ready_targets": 0,
        },
        "claim_boundary": {
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_dv_calibration_candidate_manifest_pass"
