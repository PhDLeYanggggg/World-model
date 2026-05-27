from src.stage42_source_terms_paper_claim_guard import _gate, _paper_guard_rows


def test_paper_guard_blocks_opentraj_underlying_permission_claim():
    go_payload = {
        "audit_rows": [
            {
                "dataset_id": "opentraj_toolkit",
                "terms_status": "toolkit_mit_not_underlying_dataset_terms",
                "auto_download_allowed_now": False,
                "underlying_data_license_confirmed": False,
                "contract_conversion_ready_now": False,
            }
        ]
    }
    rows = _paper_guard_rows(go_payload)
    assert rows[0]["paper_claim_status"] == "blocked_until_user_terms_path_source_confirmation"
    assert "OpenTraj" in rows[0]["allowed_paper_wording"]
    assert "underlying" in rows[0]["disallowed_paper_wording"]


def test_paper_guard_keeps_source_candidates_unconverted():
    go_payload = {
        "audit_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "auto_download_allowed_now": False,
                "underlying_data_license_confirmed": False,
                "contract_conversion_ready_now": False,
            }
        ]
    }
    rows = _paper_guard_rows(go_payload)
    assert "not counted as converted" in rows[0]["allowed_paper_wording"]
    assert "metric/seconds-calibrated" in rows[0]["disallowed_paper_wording"]


def test_gate_passes_claim_guard_without_conversion_claims():
    payload = {
        "source": "fresh_stage42_gp_source_terms_paper_claim_guard",
        "input_status": {"go_exists": True},
        "go_gate": {"passed": 14, "total": 14},
        "summary": {
            "datasets_guarded": 5,
            "paper_files_refreshed": [
                "outputs/stage42_long_research/data_card_stage42.md",
                "outputs/stage42_long_research/a_journal_gap_stage42.md",
                "outputs/stage42_long_research/method_draft_stage42.md",
            ],
            "underlying_data_license_confirmed": 0,
            "auto_download_allowed_now": 0,
            "contract_ready_now": 0,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
        "claim_scan": {"violation_count": 0},
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "restricted_metric_time_claim_allowed_now": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required_written": True,
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_gp_source_terms_paper_claim_guard_pass"
