from __future__ import annotations

import json
from pathlib import Path

from src.stage42_t50_switchability_calibration_repair import REPORT_JSON, REPORT_MD, run


def test_stage42_t50_switchability_calibration_repair_runs() -> None:
    result = run()
    gate = result["stage42_iq_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_iq_t50_switchability_calibration_repair_pass"
    assert result["horizon"] == 50
    assert len(result["trials"]) == 9


def test_stage42_t50_switchability_calibration_repair_reports_success_or_failure() -> None:
    result = json.loads(Path(REPORT_JSON).read_text())
    assert result["summary"]["verdict"] in {
        "t50_switchability_repair_supported",
        "validation_selected_gain_harm_router_still_fails_to_capture_t50_headroom",
    }
    best = result["best_trial"]
    assert "t50_improvement" in best["test_metric"]
    assert best["validation_selection"]["test_threshold_tuning"] is False


def test_stage42_t50_switchability_calibration_repair_boundaries() -> None:
    result = json.loads(Path(REPORT_JSON).read_text())
    assert result["no_leakage"]["future_endpoint_input"] is False
    assert result["no_leakage"]["future_waypoint_input"] is False
    assert result["no_leakage"]["future_label_train_val_supervision_only"] is True
    assert result["no_leakage"]["test_threshold_tuning"] is False
    assert result["claim_boundary"]["global_metric_claim_allowed"] is False
    assert result["claim_boundary"]["global_seconds_claim_allowed"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
    text = Path(REPORT_MD).read_text()
    assert "repair attempt" in text
    assert "no metric/seconds claim" in text
