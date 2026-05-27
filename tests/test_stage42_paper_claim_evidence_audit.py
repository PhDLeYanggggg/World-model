from src import stage42_paper_claim_evidence_audit as z


def _isolate_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(z, "REPORT_JSON", tmp_path / "paper_claim_evidence_audit_stage42.json")
    monkeypatch.setattr(z, "REPORT_MD", tmp_path / "paper_claim_evidence_audit_stage42.md")
    monkeypatch.setattr(z, "REPORT_CSV", tmp_path / "paper_claim_evidence_audit_stage42.csv")
    monkeypatch.setattr(z, "GATE_MD", tmp_path / "stage42_stage_z_gate.md")
    monkeypatch.setattr(z, "LEDGER_JSONL", tmp_path / "run_ledger.jsonl")


def test_stage42_z_claim_matrix_contains_negative_boundaries(tmp_path, monkeypatch):
    _isolate_outputs(tmp_path, monkeypatch)
    result = z.run_stage42_paper_claim_evidence_audit()
    statuses = {row["status"] for row in result["claim_rows"]}
    assert "supported_fresh" in statuses
    assert "rejected_by_evidence" in statuses
    assert "not_supported" in statuses
    assert "rejected_by_legal_gate" in statuses
    assert "candidate_evidence_but_claim_blocked" in statuses
    assert any(row["claim_id"] == "C9" and not row["allowed_as_main_claim"] for row in result["claim_rows"])
    assert any(row["claim_id"] == "C10" and not row["allowed_as_main_claim"] for row in result["claim_rows"])
    assert any(row["claim_id"] == "C12" and not row["allowed_as_main_claim"] for row in result["claim_rows"])
    assert any(row["claim_id"] == "C13" and not row["allowed_as_main_claim"] for row in result["claim_rows"])
    assert any(row["claim_id"] == "C14" and row["status"] == "post_confirmation_candidate_but_not_claimable" and not row["allowed_as_main_claim"] for row in result["claim_rows"])


def test_stage42_z_gate_passes_with_paper_files_and_wxy_artifacts(tmp_path, monkeypatch):
    _isolate_outputs(tmp_path, monkeypatch)
    result = z.run_stage42_paper_claim_evidence_audit()
    gate = result["stage42_z_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["gates"]["stage42x_row_cache_gate_pass"]
    assert gate["gates"]["stage42y_ablation_gate_pass"]
    assert gate["gates"]["stage42cg_source_terms_gate_pass"]
    assert gate["gates"]["stage42ch_metric_time_gate_pass"]
    assert gate["gates"]["stage42gh_calibrated_subset_gate_pass"]
    assert gate["gates"]["legal_conversion_not_overclaimed"]
    assert gate["gates"]["restricted_metric_time_not_overclaimed"]
    assert gate["gates"]["post_confirmation_candidates_not_overclaimed"]
    assert gate["gates"]["paper_files_exist"]
    assert gate["paper_ready_scope"] == "protected_2p5d_raw_frame_world_state_candidate"


def test_stage42_z_current_facts_block_overclaim():
    text = "\n".join(z.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text
