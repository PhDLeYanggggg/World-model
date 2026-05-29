from src.stage42_source_context_fresh_replay import (
    _all_nonpositive_core,
    _gate,
    run_stage42_source_context_fresh_replay,
)


def test_context_fresh_replay_records_negative_context_result():
    payload = run_stage42_source_context_fresh_replay(refresh_readmes=False)
    assert payload["summary"]["sequence_context_supported"] is False
    assert payload["summary"]["graph_context_supported"] is False
    assert payload["summary"]["baseline_family_all_improvement"] > 0
    assert payload["claim_boundary"]["sequence_context_main_claim"] is False
    assert payload["claim_boundary"]["graph_interaction_main_claim"] is False


def test_all_nonpositive_core_checks_core_metrics_only():
    rows = [
        {"all_delta": -0.1, "t50_delta": 0.0, "hard_failure_delta": -0.2, "easy_delta": 0.5},
        {"all_delta": 0.0, "t50_delta": -0.1, "hard_failure_delta": 0.0, "easy_delta": -1.0},
    ]
    assert _all_nonpositive_core(rows) is True
    assert _all_nonpositive_core(rows + [{"all_delta": 0.01, "t50_delta": -1.0, "hard_failure_delta": -1.0}]) is False


def test_jr_gate_requires_negative_result_not_overclaimed():
    payload = run_stage42_source_context_fresh_replay(refresh_readmes=False)
    gate = _gate(payload)
    assert gate["verdict"] == "stage42_jr_source_context_negative_evidence_pass"
    assert gate["gates"]["negative_result_not_overclaimed"] is True
    broken = dict(payload)
    broken["claim_boundary"] = dict(payload["claim_boundary"], graph_interaction_main_claim=True)
    broken_gate = _gate(broken)
    assert broken_gate["gates"]["negative_result_not_overclaimed"] is False
