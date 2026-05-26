from __future__ import annotations

from src.stage42_metric_time_claim_guard import _gate, _reason


def test_reason_blocks_calibrated_source_when_no_legal_readiness():
    record = {"source_specific_metric_time_evidence": True}
    assert "zero conversion-ready" in _reason(record, legal_ready_count=0)


def test_reason_blocks_incomplete_calibration():
    record = {"source_specific_metric_time_evidence": False}
    assert _reason(record, legal_ready_count=5) == "source-specific metric/time evidence incomplete"


def test_gate_blocks_global_metric_seconds_even_with_source_candidates():
    payload = {
        "source": "fresh_stage42_ch_metric_time_claim_guard",
        "summary": {
            "bn_verdict": "stage42_bn_source_time_geometry_calibration_pass_with_global_claim_blocked",
            "cg_verdict": "stage42_cg_source_terms_confirmation_validator_pass",
            "source_specific_metric_time_candidates": 6,
            "conversion_ready_targets": 0,
            "restricted_subset_metric_seconds_claim_allowed_now": False,
            "sdd_metric_claim_allowed": False,
            "tgsim_pedestrian_world_model_metric_claim_allowed": False,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required": ["x"],
    }
    gate = _gate(payload)
    assert gate["verdict"] == "stage42_ch_metric_time_claim_guard_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_global_metric_is_overclaimed():
    payload = {
        "source": "fresh_stage42_ch_metric_time_claim_guard",
        "summary": {
            "bn_verdict": "stage42_bn_source_time_geometry_calibration_pass_with_global_claim_blocked",
            "cg_verdict": "stage42_cg_source_terms_confirmation_validator_pass",
            "source_specific_metric_time_candidates": 6,
            "conversion_ready_targets": 0,
            "restricted_subset_metric_seconds_claim_allowed_now": False,
            "sdd_metric_claim_allowed": False,
            "tgsim_pedestrian_world_model_metric_claim_allowed": False,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": True,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required": ["x"],
    }
    gate = _gate(payload)
    assert gate["gates"]["global_metric_blocked"] is False
    assert gate["verdict"] == "stage42_ch_metric_time_claim_guard_partial"
