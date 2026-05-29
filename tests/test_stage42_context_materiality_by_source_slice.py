from src import stage42_context_materiality_by_source_slice as jy


def test_context_materiality_gate_passes() -> None:
    payload = jy.run_stage42_context_materiality_by_source_slice(refresh_readmes=False)
    gate = payload["stage42_jy_gate"]
    assert gate["verdict"] == "stage42_jy_context_materiality_by_source_slice_pass"
    assert gate["passed"] == gate["total"]


def test_baseline_family_remains_dominant() -> None:
    payload = jy.run_stage42_context_materiality_by_source_slice(refresh_readmes=False)
    summary = payload["summary"]
    baseline = summary["baseline_family_control"]
    assert baseline["all_improvement"] > 0
    assert baseline["t50_improvement"] > 0
    assert baseline["hard_failure_improvement"] > 0
    assert summary["material_global_incremental_variants"] == []


def test_context_claim_boundary_blocks_independent_main_claim() -> None:
    payload = jy.run_stage42_context_materiality_by_source_slice(refresh_readmes=False)
    assert payload["summary"]["context_claim_decision"] == "block_independent_context_main_claim_keep_as_auxiliary_or_new_objective"
    assert payload["claim_boundary"]["independent_context_main_claim_allowed"] is False
    assert payload["claim_boundary"]["true_3d"] is False
    assert payload["claim_boundary"]["foundation_world_model"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False


def test_next_training_spec_is_source_slice_based() -> None:
    payload = jy.run_stage42_context_materiality_by_source_slice(refresh_readmes=False)
    spec = " ".join(payload["summary"]["next_training_spec"])
    assert "source/horizon-slice objectives" in spec
    assert "Do not repeat" in spec
    assert "Stage37/teacher floor" in spec
