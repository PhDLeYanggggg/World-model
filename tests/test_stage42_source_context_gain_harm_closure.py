from src.stage42_source_context_gain_harm_closure import _gate, run_stage42_source_context_gain_harm_closure


def test_gain_harm_closure_records_t50_t100_blockers():
    payload = run_stage42_source_context_gain_harm_closure(refresh_readmes=False)
    summary = payload["summary"]
    assert summary["t50_diagnosis"]
    assert summary["t100_diagnosis"]
    assert summary["iq_repair_supported"] is False
    assert summary["ir_repair_supported"] is False
    assert summary["decision"] == "close_current_source_sequence_graph_gain_harm_family_for_t50_t100_main_claim"


def test_gain_harm_closure_preserves_narrow_horizon_evidence_without_overclaim():
    payload = run_stage42_source_context_gain_harm_closure(refresh_readmes=False)
    assert payload["summary"]["narrow_positive_horizon_routers"]
    assert payload["claim_boundary"]["sequence_graph_independent_main_claim"] is False
    assert payload["claim_boundary"]["metric_or_seconds_claim"] is False


def test_gain_harm_closure_gate_fails_if_overclaimed():
    payload = run_stage42_source_context_gain_harm_closure(refresh_readmes=False)
    gate = _gate(payload)
    assert gate["verdict"] == "stage42_js_source_context_gain_harm_closure_pass"
    broken = dict(payload)
    broken["claim_boundary"] = dict(payload["claim_boundary"], sequence_graph_independent_main_claim=True)
    broken_gate = _gate(broken)
    assert broken_gate["gates"]["negative_result_not_overclaimed"] is False
