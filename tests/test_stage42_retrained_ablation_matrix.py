from src import stage42_retrained_ablation_matrix as aa


def _isolate_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(aa, "REPORT_JSON", tmp_path / "retrained_ablation_matrix_stage42.json")
    monkeypatch.setattr(aa, "REPORT_MD", tmp_path / "retrained_ablation_matrix_stage42.md")
    monkeypatch.setattr(aa, "REPORT_CSV", tmp_path / "retrained_ablation_matrix_stage42.csv")
    monkeypatch.setattr(aa, "GATE_MD", tmp_path / "stage42_stage_aa_gate.md")
    monkeypatch.setattr(aa, "LEDGER_JSONL", tmp_path / "run_ledger.jsonl")


def test_stage42_aa_lists_all_required_ablations(tmp_path, monkeypatch):
    _isolate_outputs(tmp_path, monkeypatch)
    result = aa.run_stage42_retrained_ablation_matrix()
    names = {row["ablation"] for row in result["ablation_rows"]}
    assert set(aa.REQUIRED_ABLATIONS).issubset(names)
    gate = result["stage42_aa_gate"]
    assert gate["gates"]["all_required_ablations_listed"]


def test_stage42_aa_keeps_jepa_and_transformer_boundaries(tmp_path, monkeypatch):
    _isolate_outputs(tmp_path, monkeypatch)
    result = aa.run_stage42_retrained_ablation_matrix()
    rows = {row["ablation"]: row for row in result["ablation_rows"]}
    assert rows["no_JEPA"]["source"] == "cached_verified"
    assert rows["no_JEPA"]["main_claim_allowed"] is False
    assert rows["no_Transformer"]["variant"] == "no_transformer_proxy_history_sequence"
    assert rows["no_Transformer"]["source"] == "fresh_run"


def test_stage42_aa_gate_and_claim_boundaries(tmp_path, monkeypatch):
    _isolate_outputs(tmp_path, monkeypatch)
    result = aa.run_stage42_retrained_ablation_matrix()
    gate = result["stage42_aa_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["fresh_required_coverage"] >= 9
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
