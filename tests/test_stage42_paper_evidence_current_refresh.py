from src import stage42_paper_evidence_current_refresh as jx


def test_paper_evidence_current_refresh_passes() -> None:
    payload = jx.run_stage42_paper_evidence_current_refresh(refresh_readmes=False)
    gate = payload["stage42_jx_gate"]
    assert gate["verdict"] == "stage42_jx_current_paper_evidence_refresh_pass"
    assert gate["passed"] == gate["total"]
    assert payload["summary"]["source_slice"]["all_ade_improvement"] > 0
    assert payload["summary"]["source_slice"]["t50_ade_improvement"] > 0


def test_current_refresh_blocks_overclaims() -> None:
    payload = jx.run_stage42_paper_evidence_current_refresh(refresh_readmes=False)
    blocked = set(payload["summary"]["blocked_claims"])
    assert "JEPA_downstream_main_claim" in blocked
    assert "Transformer_independent_main_claim" in blocked
    assert "floor_free_neural_deployment" in blocked
    assert payload["claim_boundary"]["true_3d"] is False
    assert payload["claim_boundary"]["foundation_world_model"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False


def test_current_refresh_preserves_source_and_time_blockers() -> None:
    payload = jx.run_stage42_paper_evidence_current_refresh(refresh_readmes=False)
    assert payload["summary"]["source_terms"]["conversion_ready_targets"] == 0
    assert payload["summary"]["time_geometry"]["global_metric_claim_allowed"] is False
    assert payload["summary"]["time_geometry"]["global_seconds_claim_allowed"] is False


def test_current_refresh_records_teacher_floor_necessity() -> None:
    payload = jx.run_stage42_paper_evidence_current_refresh(refresh_readmes=False)
    floor = payload["summary"]["teacher_floor"]
    assert floor["fallback_rows"] > 0
    assert floor["fallback_exact_floor_rate"] >= 0.999
    assert floor["global_floor_removal_allowed"] is False
    assert floor["floor_free_neural_deployable"] is False
