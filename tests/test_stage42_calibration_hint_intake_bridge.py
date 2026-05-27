from src.stage42_calibration_hint_intake_bridge import _compact_hint, _gate, _merge_intake_with_calibration


def test_compact_hint_preserves_hint_but_blocks_claim():
    row = {
        "h_matrix_hint_count": 2,
        "time_metadata_hint_count": 1,
        "frame_stride_hint_count": 4,
        "metric_time_subset_hint": True,
        "legal_conversion_ready": False,
        "reason_claim_not_allowed": "hints_only",
        "h_matrix_hints": [{"path": "H.txt"}],
        "time_metadata_hints": [{"path": "README.md"}],
        "frame_stride_hints": [{"path": "obsmat.txt"}],
    }
    hint = _compact_hint(row)
    assert hint["hint_found"] is True
    assert hint["metric_time_subset_hint"] is True
    assert hint["claim_allowed_now"] is False
    assert hint["selected_examples"]["h_matrix"] == [{"path": "H.txt"}]


def test_merge_adds_calibration_prefill_without_confirmation():
    intake = {
        "datasets": [
            {
                "dataset_id": "ucy_crowd_original",
                "user_confirmation": {
                    "terms_accepted_by_user": False,
                    "terms_acceptance_date": "",
                    "allowed_use": "",
                    "local_path": "",
                    "source_identity": "",
                    "confirmed_by_user": "",
                },
                "conversion_ready_now": False,
            }
        ]
    }
    du = {
        "target_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "h_matrix_hint_count": 1,
                "time_metadata_hint_count": 1,
                "frame_stride_hint_count": 1,
                "metric_time_subset_hint": True,
            }
        ]
    }
    merged = _merge_intake_with_calibration(intake, du)
    row = merged["datasets"][0]
    assert row["calibration_prefill"]["hint_found"] is True
    assert row["calibration_prefill"]["claim_allowed_now"] is False
    assert row["conversion_ready_now"] is False
    assert row["user_confirmation"]["local_path"] == ""


def test_gd_gate_passes_for_safe_calibration_bridge_payload():
    payload = {
        "source": "fresh_stage42_gd_calibration_hint_intake_bridge",
        "input_status": {"du_exists": True, "intake_exists": True},
        "summary": {
            "intake_rows": 5,
            "rows_with_calibration_prefill": 5,
            "rows_with_any_calibration_hint": 3,
            "rows_with_metric_time_subset_hint": 2,
            "rows_with_user_confirmation": 0,
            "conversion_ready_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
            "metric_claim_allowed_now": False,
            "seconds_claim_allowed_now": False,
        },
        "snapshot_written": True,
        "intake_template_updated": True,
        "user_action_required_written": True,
        "claim_boundary": {
            "download_executed": False,
            "conversion_executed": False,
            "evaluation_executed": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_gd_calibration_hint_intake_bridge_pass"
