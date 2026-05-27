from __future__ import annotations

from src import stage42_objective_coverage_audit as fx


def test_coverage_summary_marks_goal_incomplete() -> None:
    rows = [
        {"objective_id": "A", "status": "blocked_user_action_required"},
        {"objective_id": "B", "status": "partial_positive_with_source_blockers"},
        {"objective_id": "E", "status": "pass_floor_required"},
    ]

    summary = fx._coverage_summary(rows)

    assert summary["goal_complete"] is False
    assert summary["blocked_objectives"] == ["A"]
    assert summary["partial_objectives"] == ["B"]
    assert summary["passed_objectives"] == ["E"]


def test_objective_rows_cover_stage42_a_to_f() -> None:
    rows = fx._objective_rows(
        {
            "module_ledger": {
                "summary": {
                    "main_claim_allowed_modules": ["history", "teacher_floor"],
                    "blocked_or_auxiliary_modules": ["JEPA", "Transformer"],
                }
            },
            "source_action": {"summary": {"top_actions": ["FW-TERMS-ucy_crowd_original"]}},
        }
    )

    assert [row["objective_id"] for row in rows] == list("ABCDEF")
    assert rows[0]["status"] == "blocked_user_action_required"
    assert any("JEPA" in item for item in rows[3]["missing"])


def test_gate_passes_for_complete_audit_payload() -> None:
    rows = [
        {
            "objective_id": objective_id,
            "status": status,
            "evidence": ["x.json"],
            "missing": ["gap"],
            "next_actions": ["next"],
        }
        for objective_id, status in [
            ("A", "blocked_user_action_required"),
            ("B", "partial_positive_with_source_blockers"),
            ("C", "partial_protected_not_ungated"),
            ("D", "partial_main_modules_identified"),
            ("E", "pass_floor_required"),
            ("F", "paper_package_candidate_clean_with_open_blockers"),
        ]
    ]
    payload = {
        "source": fx.SOURCE,
        "objective_rows": rows,
        "summary": fx._coverage_summary(rows),
        "inputs": {"source_action": {"stage42_fw_gate": {"passed": 16, "total": 16}, "summary": {"conversion_ready_now": 0}}},
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "threshold_tuned_on_test": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fx._gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_fx_objective_coverage_audit_pass"


def test_gate_fails_if_goal_marked_complete() -> None:
    rows = [
        {"objective_id": c, "status": "pass", "evidence": ["x"], "missing": ["gap"], "next_actions": ["next"]}
        for c in "ABCDEF"
    ]
    summary = fx._coverage_summary(rows)
    summary["goal_complete"] = True
    payload = {
        "source": fx.SOURCE,
        "objective_rows": rows,
        "summary": summary,
        "inputs": {"source_action": {"stage42_fw_gate": {"passed": 16, "total": 16}, "summary": {"conversion_ready_now": 0}}},
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "threshold_tuned_on_test": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fx._gate(payload)

    assert gate["gates"]["goal_not_marked_complete"] is False
    assert gate["verdict"] == "stage42_fx_objective_coverage_audit_partial"
