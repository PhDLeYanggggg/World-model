from __future__ import annotations

from src.stage42_long_objective_coverage_audit import _gate, _summary


def _rows() -> list[dict]:
    return [
        {
            "phase": "A data and calibration",
            "requirement_id": "A1",
            "status": "partial_blocked",
            "blockers": ["source_terms_confirmation_missing"],
        },
        {"phase": "B external validation", "requirement_id": "B1", "status": "pass_with_boundary", "blockers": []},
        {"phase": "C full-waypoint dynamics", "requirement_id": "C1", "status": "pass_with_boundary", "blockers": []},
        {
            "phase": "D causal ablation",
            "requirement_id": "D1",
            "status": "mixed",
            "blockers": ["scene_goal_main_claim_blocked"],
        },
        {
            "phase": "E safety floor",
            "requirement_id": "E1",
            "status": "pass_with_boundary",
            "blockers": ["ungated_neural_not_deployable"],
        },
        {
            "phase": "F paper package",
            "requirement_id": "F1",
            "status": "pass_with_open_gaps",
            "blockers": ["source_terms_open"],
        },
        {"phase": "A data and calibration", "requirement_id": "A2", "status": "partial_blocked", "blockers": []},
    ]


def test_stage42_ek_summary_preserves_open_blockers_and_disallows_completion() -> None:
    summary = _summary(_rows())

    assert summary["requirements_audited"] == 7
    assert "A data and calibration" in summary["phases_audited"]
    assert summary["completion_claim_allowed"] is False
    assert summary["a_journal_ready_claim_allowed"] is False
    assert "source_terms_confirmation_missing" in summary["open_blockers"]


def test_stage42_ek_gate_requires_all_phases_and_no_overclaim() -> None:
    summary = _summary(_rows())
    summary["paper_files_present"] = summary["paper_files_total"]
    payload = {
        "summary": summary,
        "claim_boundary": {
            "conversion_executed_in_stage42_ek": False,
            "evaluation_executed_in_stage42_ek": False,
            "training_executed_in_stage42_ek": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ek_long_objective_coverage_audit_pass_open_blockers"
