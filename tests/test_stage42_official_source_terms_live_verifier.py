from src.stage42_official_source_terms_live_verifier import _audit_rows, _gate


def test_opentraj_toolkit_license_is_not_underlying_dataset_permission():
    priority = {"priority_rows": [{"dataset_id": "opentraj_toolkit", "priority_rank": 1, "domain": "OpenTraj"}]}
    contract = {"contract_rows": [{"dataset_id": "opentraj_toolkit", "contract_conversion_ready_now": False}]}
    rows = _audit_rows(priority, contract)
    row = [item for item in rows if item["dataset_id"] == "opentraj_toolkit"][0]
    assert row["terms_status"] == "toolkit_mit_not_underlying_dataset_terms"
    assert row["underlying_data_license_confirmed"] is False
    assert row["auto_download_allowed_now"] is False


def test_eth_live_status_does_not_auto_accept_terms():
    priority = {"priority_rows": [{"dataset_id": "eth_biwi_original", "priority_rank": 1, "domain": "ETH_UCY"}]}
    contract = {"contract_rows": [{"dataset_id": "eth_biwi_original", "contract_conversion_ready_now": False}]}
    rows = _audit_rows(priority, contract)
    row = [item for item in rows if item["dataset_id"] == "eth_biwi_original"][0]
    assert row["official_source_live_status"] == "official_page_reachable_with_dataset_download_links"
    assert row["terms_status"] == "not_verified_by_agent"
    assert row["contract_conversion_ready_now"] is False


def test_gate_requires_no_auto_download_or_conversion():
    payload = {
        "source": "fresh_stage42_go_official_source_terms_live_verifier",
        "input_status": {"priority_exists": True, "contract_exists": True},
        "audit_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "official_url": "https://example.org",
                "user_must_confirm": ["terms"],
            }
            for _ in range(5)
        ],
        "summary": {
            "datasets_audited": 5,
            "underlying_data_license_confirmed": 0,
            "contract_ready_now": 0,
            "auto_download_allowed_now": 0,
            "total_t50_after_terms": 1,
            "total_t100_after_terms": 1,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
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
    assert gate["verdict"] == "stage42_go_official_source_terms_live_verifier_pass"
