from src.stage42_long_objective_state_reconciler import CLAIM_BOUNDARY, _gate, _objective_rows, _summary


def test_objective_rows_preserve_source_blocker_when_contract_not_ready():
    source_summary = {
        "contract_ready_now": 0,
        "package_source_claim_violations": 0,
    }
    rows = _objective_rows(source_summary)
    assert len(rows) == 6
    by_name = {row["objective"]: row for row in rows}
    assert by_name["A_data_and_calibration"]["status"] == "blocked_user_action_required"
    assert by_name["F_paper_package"]["status"] == "claim_safe_with_open_data_blocker"
    assert {row["result_source"] for row in rows} == {"fresh_run", "cached_verified"}


def test_gate_passes_for_clean_reconciled_state():
    source_summary = {
        "contract_ready_now": 0,
        "auto_download_allowed_now": 0,
        "package_source_claim_violations": 0,
        "after_terms_t50_opportunity": 10060,
        "after_terms_t100_opportunity": 5696,
    }
    rows = _objective_rows({"contract_ready_now": 0, "package_source_claim_violations": 0})
    payload = {
        "source": "fresh_stage42_gr_long_objective_state_reconciler",
        "input_status": {
            name: {"exists": True, "gate_passed": True}
            for name in [
                "gl_contract",
                "gm_harness",
                "gn_priority",
                "go_live_terms",
                "gp_claim_guard",
                "gq_package_linter",
            ]
        },
        "summary": _summary({}, source_summary, rows),
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required": ["confirm terms"],
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_gr_long_objective_state_reconciler_pass"


def test_gate_fails_if_stage5c_or_smc_enabled():
    source_summary = {
        "contract_ready_now": 0,
        "auto_download_allowed_now": 0,
        "package_source_claim_violations": 0,
        "after_terms_t50_opportunity": 10060,
        "after_terms_t100_opportunity": 5696,
    }
    rows = _objective_rows({"contract_ready_now": 0, "package_source_claim_violations": 0})
    boundary = dict(CLAIM_BOUNDARY)
    boundary["stage5c_executed"] = True
    boundary["smc_enabled"] = True
    payload = {
        "source": "fresh_stage42_gr_long_objective_state_reconciler",
        "input_status": {
            name: {"exists": True, "gate_passed": True}
            for name in [
                "gl_contract",
                "gm_harness",
                "gn_priority",
                "go_live_terms",
                "gp_claim_guard",
                "gq_package_linter",
            ]
        },
        "summary": _summary({}, source_summary, rows),
        "claim_boundary": boundary,
        "user_action_required": ["confirm terms"],
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"] - 2
    assert gate["gates"]["stage5c_not_executed"] is False
    assert gate["gates"]["smc_not_enabled"] is False
