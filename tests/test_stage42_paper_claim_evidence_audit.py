from src import stage42_paper_claim_evidence_audit as z


def test_stage42_z_claim_matrix_contains_negative_boundaries():
    result = z.run_stage42_paper_claim_evidence_audit()
    statuses = {row["status"] for row in result["claim_rows"]}
    assert "supported_fresh" in statuses
    assert "rejected_by_evidence" in statuses
    assert "not_supported" in statuses
    assert any(row["claim_id"] == "C9" and not row["allowed_as_main_claim"] for row in result["claim_rows"])
    assert any(row["claim_id"] == "C10" and not row["allowed_as_main_claim"] for row in result["claim_rows"])


def test_stage42_z_gate_passes_with_paper_files_and_wxy_artifacts():
    result = z.run_stage42_paper_claim_evidence_audit()
    gate = result["stage42_z_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["gates"]["stage42x_row_cache_gate_pass"]
    assert gate["gates"]["stage42y_ablation_gate_pass"]
    assert gate["gates"]["paper_files_exist"]
    assert gate["paper_ready_scope"] == "protected_2p5d_raw_frame_world_state_candidate"


def test_stage42_z_current_facts_block_overclaim():
    text = "\n".join(z.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text
