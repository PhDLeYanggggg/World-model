from src.stage42_prefill_intake_bridge import _gate, _has_user_confirmation, _merge_intake_with_prefill


def test_merge_adds_prefill_without_filling_user_confirmation():
    intake = {
        "source": "fresh_source_terms_confirmation_intake_from_stage42_ef",
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
        ],
    }
    prefill = {
        "datasets": [
            {
                "dataset_id": "ucy_crowd_original",
                "suggested_local_path": "external_data/OpenTraj/datasets/UCY",
                "suggested_source_identity": "UCY local candidate",
                "local_path_candidates": [{"path": "external_data/OpenTraj/datasets/UCY"}],
                "safe_copy_instruction": "copy only after terms",
            }
        ]
    }
    merged = _merge_intake_with_prefill(intake, prefill)
    row = merged["datasets"][0]
    assert row["prefill_suggestion"]["suggested_local_path"] == "external_data/OpenTraj/datasets/UCY"
    assert row["prefill_suggestion"]["agent_may_copy_without_user_terms_confirmation"] is False
    assert _has_user_confirmation(row) is False
    assert row["conversion_ready_now"] is False


def test_has_user_confirmation_detects_manual_fields():
    row = {
        "user_confirmation": {
            "terms_accepted_by_user": False,
            "terms_acceptance_date": "",
            "allowed_use": "research_only",
            "local_path": "",
            "source_identity": "",
            "confirmed_by_user": "",
        }
    }
    assert _has_user_confirmation(row) is True


def test_gc_gate_passes_for_safe_bridge_payload():
    payload = {
        "source": "fresh_stage42_gc_prefill_intake_bridge",
        "input_status": {"prefill_exists": True, "intake_exists": True},
        "summary": {
            "intake_rows": 5,
            "rows_with_prefill_suggestion": 5,
            "rows_with_user_confirmation": 0,
            "conversion_ready_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
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
    assert gate["verdict"] == "stage42_gc_prefill_intake_bridge_pass"
