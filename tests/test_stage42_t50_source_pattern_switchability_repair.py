from __future__ import annotations

import json
from pathlib import Path

from src.stage42_t50_source_pattern_switchability_repair import REPORT_JSON, REPORT_MD, run


def test_stage42_t50_source_pattern_switchability_repair_runs() -> None:
    result = run()
    gate = result["stage42_ir_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ir_t50_source_pattern_switchability_repair_pass"
    assert result["source_pattern_schema"]["stats"]["feature_count"] >= 8
    assert len(result["trials"]) == 9


def test_stage42_t50_source_pattern_switchability_repair_reports_boundary() -> None:
    result = json.loads(Path(REPORT_JSON).read_text())
    assert result["summary"]["verdict"] in {
        "t50_source_pattern_switchability_repair_supported",
        "t50_source_pattern_switchability_repair_not_supported",
    }
    assert result["no_leakage"]["source_file_pattern_input_only"] is True
    assert result["no_leakage"]["future_endpoint_input"] is False
    assert result["no_leakage"]["future_waypoint_input"] is False
    assert result["claim_boundary"]["global_metric_claim_allowed"] is False
    assert result["claim_boundary"]["global_seconds_claim_allowed"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
    assert "source-pattern repair attempt" in Path(REPORT_MD).read_text()
