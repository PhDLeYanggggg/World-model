from src import stage42_full_waypoint_claim_guard as gz


def _isolate_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gz, "REPORT_JSON", tmp_path / "full_waypoint_claim_guard_stage42.json")
    monkeypatch.setattr(gz, "REPORT_MD", tmp_path / "full_waypoint_claim_guard_stage42.md")
    monkeypatch.setattr(gz, "GATE_MD", tmp_path / "stage42_stage_gz_gate.md")


def test_stage42_gz_gate_passes_and_blocks_overclaims(tmp_path, monkeypatch):
    _isolate_outputs(tmp_path, monkeypatch)
    result = gz.run_stage42_full_waypoint_claim_guard()
    gate = result["stage42_gz_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["gates"]["protected_full_waypoint_positive_recorded"]
    assert gate["gates"]["linear_bridge_all_advantage_preserved"]
    assert gate["gates"]["ungated_full_waypoint_blocked"]
    assert gate["gates"]["endpoint_as_full_waypoint_blocked"]
    assert gate["gates"]["global_primary_replacement_blocked"]
    assert gate["gates"]["group_consistency_allowed_only_with_boundary"]
    assert gate["gates"]["no_metric_seconds_overclaim"]
    assert gate["gates"]["stage5c_false"]
    assert gate["gates"]["smc_false"]


def test_stage42_gz_claim_rows_have_allowed_and_rejected_boundaries(tmp_path, monkeypatch):
    _isolate_outputs(tmp_path, monkeypatch)
    result = gz.run_stage42_full_waypoint_claim_guard()
    rows = {row["claim_id"]: row for row in result["claim_rows"]}
    assert rows["GZ-C1"]["allowed_as_main_claim"]
    assert rows["GZ-C4"]["allowed_as_main_claim"]
    assert rows["GZ-C6"]["allowed_as_main_claim"]
    assert rows["GZ-C2"]["allowed_as_main_claim"] is False
    assert rows["GZ-C3"]["allowed_as_main_claim"] is False
    assert rows["GZ-C5"]["allowed_as_main_claim"] is False
    assert rows["GZ-C8"]["allowed_as_main_claim"] is False
    assert rows["GZ-C9"]["allowed_as_main_claim"] is False
    assert "raw-frame" in rows["GZ-C1"]["required_boundary"]
    assert "endpoint" in rows["GZ-C2"]["required_boundary"]


def test_stage42_gz_current_facts_keep_stage5c_and_smc_disabled():
    facts = "\n".join(gz.CURRENT_FACTS)
    assert "不是 true 3D" in facts
    assert "raw-frame" in facts
    assert "Stage5C" in facts
    assert "SMC" in facts
