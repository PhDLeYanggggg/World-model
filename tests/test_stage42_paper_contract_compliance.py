from src.stage42_paper_contract_compliance import _context_is_bounded, _gate, _overclaim_hits


def _payload(**summary_overrides):
    summary = {
        "contract_gate_passed": True,
        "paper_files_total": 9,
        "paper_files_existing": 9,
        "paper_files_with_dataset_local": 9,
        "paper_files_with_raw_frame": 9,
        "paper_files_with_2_5d": 9,
        "paper_files_with_stage5c_boundary": 9,
        "paper_files_with_smc_boundary": 9,
        "unbounded_overclaim_hits": 0,
        "supported_anchor_count": 5,
        "supported_anchor_covered": 5,
        "blocked_claim_count": 7,
        "blocked_claims_covered_as_limitation": 7,
        "new_training_or_conversion": False,
    }
    summary.update(summary_overrides)
    return {
        "summary": summary,
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_bounded_context_does_not_count_negative_true3d_statement():
    text = "Current model is not a true 3D world model and foundation claims are blocked."
    assert _context_is_bounded(text)
    assert _overclaim_hits(text) == []


def test_unbounded_foundation_claim_is_flagged():
    text = "M3W is a foundation world model for global deployment."
    hits = _overclaim_hits(text)
    assert hits
    assert hits[0]["claim_family"] == "foundation"


def test_gate_passes_complete_compliance_payload():
    gate = _gate(_payload())
    assert gate["verdict"] == "stage42_ie_paper_contract_compliance_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_on_unbounded_overclaim():
    gate = _gate(_payload(unbounded_overclaim_hits=1))
    assert gate["gates"]["no_unbounded_overclaims"] is False


def test_gate_fails_when_supported_anchor_missing():
    gate = _gate(_payload(supported_anchor_covered=4))
    assert gate["gates"]["supported_anchors_covered"] is False
