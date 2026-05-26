from __future__ import annotations

from src.stage42_source_conversion_legal_gate import _confirmation_template, _decide_target, _gate


def _row(**overrides):
    row = {
        "id": "ucy_crowd_original",
        "name": "UCY Crowd Data",
        "priority": "critical",
        "official_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
        "local_path_found": True,
        "schema_possible": True,
        "t50_capable_files": 2,
        "t100_capable_files": 2,
        "independent_t50_candidate_files": 1,
        "legal_terms_blocked": True,
        "source_cv_preflight_ready": False,
    }
    row.update(overrides)
    return row


def test_terms_block_conversion_even_when_schema_and_t50_exist():
    decision = _decide_target(_row())
    assert decision["conversion_allowed_now"] is False
    assert "manual_terms_or_application_required" in decision["blockers"]
    assert decision["converted_now"] is False
    assert decision["evaluated_now"] is False


def test_alternate_current_source_blocks_conversion_without_independent_t50():
    decision = _decide_target(_row(legal_terms_blocked=False, independent_t50_candidate_files=0))
    assert decision["conversion_allowed_now"] is False
    assert "no_independent_t50_candidate" in decision["blockers"]


def test_confirmation_template_is_not_permission():
    decisions = [_decide_target(_row())]
    template = _confirmation_template(decisions)
    assert template["terms_confirmation_is_currently_absent"] is True
    assert template["datasets"][0]["terms_accepted_by_user"] is False
    assert template["datasets"][0]["allowed_use"] == ""


def test_gate_rejects_conversion_overclaim():
    decision = _decide_target(_row())
    payload = {
        "source": "fresh_stage42_cf_source_conversion_legal_gate",
        "input_reports": {"stage42_ce_verdict": "stage42_ce_source_diversity_conversion_preflight_pass"},
        "summary": {
            "conversion_allowed_now_count": 1,
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "source_cv_ready_now": 0,
        },
        "target_decisions": [decision],
        "confirmation_template": {"terms_confirmation_is_currently_absent": True},
        "user_action_required": [{"target": "UCY"}],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = _gate(payload)
    assert gate["gates"]["no_conversion_allowed_without_confirmation"] is False
    assert gate["verdict"] == "stage42_cf_source_conversion_legal_gate_partial"
