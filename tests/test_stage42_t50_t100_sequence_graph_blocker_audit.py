from __future__ import annotations

import json
from pathlib import Path

from src.stage42_t50_t100_sequence_graph_blocker_audit import REPORT_JSON, REPORT_MD, run


def test_stage42_t50_t100_sequence_graph_blocker_audit_runs() -> None:
    result = run()
    gate = result["stage42_ip_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ip_t50_t100_sequence_graph_blocker_audit_pass"
    assert set(result["target_horizons"]) == {50, 100}
    assert len(result["candidate_horizon_audits"]) == 6


def test_stage42_t50_t100_sequence_graph_blocker_audit_has_diagnoses() -> None:
    result = json.loads(Path(REPORT_JSON).read_text())
    assert result["summary"]["t50_diagnosis"]
    assert result["summary"]["t100_diagnosis"]
    for row in result["candidate_horizon_audits"].values():
        assert row["oracle_headroom"] >= 0.0
        assert 0.0 <= row["capture_rate"] <= 1.0 + 1e-6
        assert row["blocker_diagnosis"] in {
            "candidate_oracle_headroom_too_small",
            "low_margin_candidate_ambiguity",
            "train_test_distribution_shift",
            "router_under_switches_despite_headroom",
            "unsafe_or_uncalibrated_switching",
            "weak_predictive_signal_or_baseline_family_dominance",
        }


def test_stage42_t50_t100_sequence_graph_blocker_audit_boundaries() -> None:
    result = json.loads(Path(REPORT_JSON).read_text())
    assert result["no_leakage"]["future_endpoint_input"] is False
    assert result["no_leakage"]["future_waypoint_input"] is False
    assert result["no_leakage"]["test_threshold_tuning"] is False
    assert result["claim_boundary"]["global_metric_claim_allowed"] is False
    assert result["claim_boundary"]["global_seconds_claim_allowed"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
    text = Path(REPORT_MD).read_text()
    assert "no metric/seconds claim" in text
