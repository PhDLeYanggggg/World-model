from src.stage42_paper_gap_reconciler import CLAIM_BOUNDARY, _gate, _stale_gap_findings


def test_stale_gap_findings_detect_superseded_stage42_p_language():
    text = """
    Stage42-P repairs the mean ADE t50 sign, but the 3-seed t50 CI low remains negative.
    Build a proximity-safe internal self-gate that reduces teacher-floor dependence.
    """
    findings = _stale_gap_findings(text)
    ids = {row["finding_id"] for row in findings}
    assert "stage42_p_t50_ci_low_negative" in ids
    assert "proximity_self_gate_open" in ids


def test_gate_passes_for_clean_reconciled_gap_package():
    payload = {
        "source": "fresh_stage42_gs_paper_gap_reconciler",
        "input_status": {
            name: {"exists": True}
            for name in ["module_ledger", "floor_map", "long_objective", "package_linter", "gap_md"]
        },
        "summary": {
            "gap_rows": 5,
            "stale_findings_reconciled": 2,
            "open_blockers": ["source_legal_conversion", "floor_free_neural_deployment"],
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
        "gap_rows": [
            {"gap": "module_contribution_boundary"},
            {"gap": "paper_package_source_claim_safety", "current_value": {"violation_count": 0}},
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_gs_paper_gap_reconciler_pass"


def test_gate_blocks_metric_or_stage5c_overclaim():
    boundary = dict(CLAIM_BOUNDARY)
    boundary["global_metric_claim_allowed"] = True
    boundary["stage5c_executed"] = True
    payload = {
        "source": "fresh_stage42_gs_paper_gap_reconciler",
        "input_status": {
            name: {"exists": True}
            for name in ["module_ledger", "floor_map", "long_objective", "package_linter", "gap_md"]
        },
        "summary": {
            "gap_rows": 5,
            "stale_findings_reconciled": 2,
            "open_blockers": ["source_legal_conversion", "floor_free_neural_deployment"],
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
        "gap_rows": [
            {"gap": "module_contribution_boundary"},
            {"gap": "paper_package_source_claim_safety", "current_value": {"violation_count": 0}},
        ],
        "claim_boundary": boundary,
    }
    gate = _gate(payload)
    assert gate["gates"]["claim_boundary_no_metric_seconds"] is False
    assert gate["gates"]["stage5c_not_executed"] is False
