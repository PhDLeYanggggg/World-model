from pathlib import Path

from src.stage42_paper_package_fxfy_refresh import _fxfy_summary, _gate, _scan_overclaims


def _gate_block(passed: int = 1, total: int = 1, verdict: str = "pass"):
    return {"passed": passed, "total": total, "verdict": verdict}


def test_fxfy_summary_preserves_blockers_and_claims():
    inputs = {
        "module_ledger": {
            "summary": {
                "paper_claim_core": ["history", "safe_switch"],
                "paper_claim_blocked": ["JEPA_downstream_lift"],
            }
        },
        "source_action": {
            "summary": {
                "conversion_ready_now": 0,
                "claim_ready_after_this_stage": False,
                "highest_priority_blocker": "FW-TERMS-ucy_crowd_original",
            }
        },
        "objective_coverage": {
            "summary": {
                "status_counts": {"blocked_user_action_required": 1},
                "blocked_objectives": ["A"],
                "partial_objectives": ["B"],
                "passed_objectives": ["E"],
                "goal_complete": False,
            }
        },
        "horizon_retry": {
            "summary": {
                "weak_horizons": ["TrajNet|100", "UCY|100"],
                "stop_repeat_modeling_now": True,
                "uniform_horizon_claim_allowed": False,
            }
        },
    }
    summary = _fxfy_summary(inputs)
    assert summary["supported_core_claims"] == ["history", "safe_switch"]
    assert summary["blocked_objectives"] == ["A"]
    assert summary["weak_horizons"] == ["TrajNet|100", "UCY|100"]
    assert summary["stop_repeat_modeling_now"] is True
    assert summary["uniform_horizon_claim_allowed"] is False


def test_fz_gate_passes_for_complete_refresh_payload():
    payload = {
        "source": "fresh_stage42_paper_package_fxfy_refresh",
        "inputs": {
            "module_ledger": {"stage42_fu_gate": _gate_block()},
            "claim_linter": {"stage42_fv_gate": _gate_block()},
            "source_action": {"stage42_fw_gate": _gate_block()},
            "reviewer_replay": {"stage42_dm_gate": _gate_block()},
            "objective_coverage": {"stage42_fx_gate": _gate_block()},
            "horizon_retry": {"stage42_fy_gate": _gate_block()},
        },
        "summary": {
            "goal_complete": False,
            "blocked_objectives": ["A"],
            "weak_horizons": ["TrajNet|100", "UCY|100"],
            "stop_repeat_modeling_now": True,
            "uniform_horizon_claim_allowed": False,
        },
        "paper_file_status": [{"contains_fz_marker": True}, {"contains_fz_marker": True}],
        "overclaim_violations": [],
        "claim_boundary": {
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "threshold_tuned_on_test": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_fz_paper_package_fxfy_refresh_pass"


def test_fz_gate_fails_when_uniform_horizon_is_claimed():
    payload = {
        "source": "fresh_stage42_paper_package_fxfy_refresh",
        "inputs": {
            "module_ledger": {"stage42_fu_gate": _gate_block()},
            "claim_linter": {"stage42_fv_gate": _gate_block()},
            "source_action": {"stage42_fw_gate": _gate_block()},
            "reviewer_replay": {"stage42_dm_gate": _gate_block()},
            "objective_coverage": {"stage42_fx_gate": _gate_block()},
            "horizon_retry": {"stage42_fy_gate": _gate_block()},
        },
        "summary": {
            "goal_complete": False,
            "blocked_objectives": ["A"],
            "weak_horizons": ["TrajNet|100", "UCY|100"],
            "stop_repeat_modeling_now": True,
            "uniform_horizon_claim_allowed": True,
        },
        "paper_file_status": [{"contains_fz_marker": True}],
        "overclaim_violations": [],
        "claim_boundary": {
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "threshold_tuned_on_test": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = _gate(payload)
    assert gate["gates"]["uniform_horizon_not_claimed"] is False
    assert gate["verdict"] == "stage42_fz_paper_package_fxfy_refresh_partial"


def test_scan_overclaims_detects_forbidden_phrase(tmp_path: Path):
    path = tmp_path / "paper.md"
    path.write_text("This is a true 3D world model.", encoding="utf-8")
    violations = _scan_overclaims([path])
    assert violations
    assert violations[0]["check"] == "true_3d"
