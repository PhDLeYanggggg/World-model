from src.stage42_source_conversion_contract import _confirmation_status, _contract_rows, _gate, _summary


def _intake_row(**overrides):
    row = {
        "priority_rank": 1,
        "dataset_id": "ucy_crowd_original",
        "domain": "UCY",
        "official_url_from_prior_audit": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
        "after_terms_potential": {"estimated_t50_windows": 10, "estimated_t100_windows": 5},
        "agent_may_fill": False,
        "user_confirmation": {
            "terms_accepted_by_user": False,
            "terms_acceptance_date": "",
            "official_terms_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
            "accepted_terms_version_or_access_date": "",
            "allowed_use": "",
            "redistribution_allowed": "unknown",
            "derived_data_allowed": "unknown",
            "local_path": "",
            "source_identity": "",
            "confirmed_by_user": "",
        },
    }
    row.update(overrides)
    return row


def test_blank_confirmation_is_not_complete():
    status = _confirmation_status(_intake_row())
    assert status["all_required_fields_filled"] is False
    assert "terms_accepted_by_user" in status["missing_fields"]
    assert "local_path" in status["missing_fields"]


def test_contract_blocks_blank_intake_even_with_calibrated_candidates():
    intake = {"datasets": [_intake_row()]}
    manifest = {
        "conversion_ready_targets": [],
        "blocked_targets": [
            {
                "dataset_id": "ucy_crowd_original",
                "conversion_ready": False,
                "confirmation_blockers": ["terms_not_accepted"],
                "cf_blockers": ["manual_terms_or_application_required"],
            }
        ],
    }
    gh = {
        "plan_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "domain": "UCY",
                "source_id": "UCY_zara01",
                "restricted_metric_time_candidate_after_terms": True,
                "restricted_metric_time_ready_now": False,
                "t50_windows_after_terms": 100,
                "t100_windows_after_terms": 50,
                "allowed_local_claim_after_legal_conversion": "source_specific_annotation_step_meter_coordinate_evidence",
            }
        ],
        "summary": {"restricted_metric_time_candidates_after_terms": 1},
    }
    rows = _contract_rows(intake, manifest, gh)
    assert rows[0]["contract_conversion_ready_now"] is False
    assert rows[0]["contract_status"] == "blocked_until_user_terms_path_source_confirmation"
    assert rows[0]["calibrated_subset_after_terms"]["calibrated_t50_windows_after_terms"] == 100


def test_gate_passes_honest_no_ready_contract():
    rows = _contract_rows(
        {"datasets": [_intake_row()]},
        {"conversion_ready_targets": [], "blocked_targets": [{"dataset_id": "ucy_crowd_original"}]},
        {
            "plan_rows": [
                {
                    "dataset_id": "ucy_crowd_original",
                    "restricted_metric_time_candidate_after_terms": True,
                    "restricted_metric_time_ready_now": False,
                    "t50_windows_after_terms": 10,
                    "t100_windows_after_terms": 5,
                }
            ],
            "summary": {"restricted_metric_time_candidates_after_terms": 1},
        },
    )
    summary = _summary(
        rows,
        {"conversion_ready_targets": [], "blocked_targets": [{"dataset_id": "ucy_crowd_original"}]},
        {"summary": {"restricted_metric_time_candidates_after_terms": 1}},
        {"conversion_queue": []},
    )
    payload = {
        "source": "fresh_stage42_gl_source_conversion_contract",
        "input_status": {"intake_exists": True, "manifest_exists": True, "gh_exists": True, "ej_exists": True},
        "contract_rows": rows,
        "summary": summary,
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "restricted_subset_claim_allowed_now": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required_written": True,
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_gl_source_conversion_contract_pass"
