from pathlib import Path

from src import stage42_full_waypoint_overclaim_linter as ha


def _isolate_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ha, "REPORT_JSON", tmp_path / "full_waypoint_overclaim_linter_stage42.json")
    monkeypatch.setattr(ha, "REPORT_MD", tmp_path / "full_waypoint_overclaim_linter_stage42.md")
    monkeypatch.setattr(ha, "GATE_MD", tmp_path / "stage42_stage_ha_gate.md")


def test_scan_file_flags_endpoint_as_full_waypoint_overclaim(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("Endpoint-linear bridge is learned full-waypoint dynamics.\n", encoding="utf-8")

    rows = ha.scan_file(path)

    assert rows
    assert rows[0]["check"] == "endpoint_as_full_waypoint"


def test_scan_file_allows_negated_full_waypoint_boundary(tmp_path: Path) -> None:
    path = tmp_path / "ok.md"
    path.write_text(
        "# Claim Boundary\n\nEndpoint-only bridge must not be counted as learned full-waypoint dynamics.\n",
        encoding="utf-8",
    )

    assert ha.scan_file(path) == []


def test_stage42_ha_gate_passes_current_paper_package(tmp_path, monkeypatch):
    _isolate_outputs(tmp_path, monkeypatch)
    result = ha.run_stage42_full_waypoint_overclaim_linter()
    gate = result["stage42_ha_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["gates"]["gz_claim_guard_loaded"]
    assert gate["gates"]["gj_module_claim_lock_loaded"]
    assert gate["gates"]["no_endpoint_as_full_waypoint_overclaim"]
    assert gate["gates"]["no_ungated_full_waypoint_deployable_overclaim"]
    assert gate["gates"]["no_global_primary_replacement_overclaim"]
    assert gate["gates"]["no_neighbor_interaction_independent_overclaim"]
    assert gate["gates"]["no_metric_seconds_or_3d_overclaim"]
    assert gate["gates"]["stage5c_false"]
    assert gate["gates"]["smc_false"]


def test_stage42_ha_current_facts_keep_boundaries():
    facts = "\n".join(ha.CURRENT_FACTS)
    assert "不是 true 3D" in facts
    assert "endpoint" in facts
    assert "Stage5C" in facts
    assert "SMC" in facts
