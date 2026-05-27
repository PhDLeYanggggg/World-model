from src.stage42_floor_relaxation_paper_refresh import run_stage42_floor_relaxation_paper_refresh


def test_stage42_floor_relaxation_paper_refresh_outputs():
    payload = run_stage42_floor_relaxation_paper_refresh()
    assert payload["source"] == "fresh_stage42_gu_floor_relaxation_paper_refresh"
    assert payload["stage42_gu_gate"]["passed"] == payload["stage42_gu_gate"]["total"]
    assert payload["summary"]["target_union_rows"] > 0
    assert payload["summary"]["target_union_t50_improvement"] > 0.0
    assert payload["summary"]["floor_claim_violation_count"] == 0
    assert payload["claim_boundary"]["partial_t50_floor_relaxation_allowed"] is True
    assert payload["claim_boundary"]["global_floor_removal_allowed"] is False
    assert payload["claim_boundary"]["floor_free_neural_deployable"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
