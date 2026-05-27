from __future__ import annotations

from src import stage42_module_contribution_ledger as fu


def test_pick_metrics_and_deltas_are_claim_safe() -> None:
    row = {
        "all": 0.1,
        "t50": 0.2,
        "hard_failure": 0.3,
        "delta_all_full_minus_ablation": 0.4,
        "delta_t50_full_minus_ablation": 0.5,
        "unused": "x",
    }

    assert fu._pick_metrics(row) == {"all": 0.1, "t50": 0.2, "hard_failure": 0.3}
    assert fu._pick_deltas(row) == {"delta_all": 0.4, "delta_t50": 0.5}


def test_summary_separates_main_and_blocked_claims() -> None:
    rows = [
        {"module": "history", "status": "supported_main_claim", "main_claim_allowed": True},
        {"module": "JEPA", "status": "blocked_negative_or_inconclusive", "main_claim_allowed": False},
    ]

    summary = fu._summary(rows)

    assert summary["supported_or_necessary_modules"] == 1
    assert summary["main_claim_allowed_modules"] == ["history"]
    assert summary["blocked_or_auxiliary_modules"] == ["JEPA"]
    assert "JEPA_downstream_lift" in summary["paper_claim_blocked"]


def test_gate_requires_core_supported_and_overclaims_blocked() -> None:
    payload = {
        "source": fu.SOURCE,
        "input_gates": {
            "aa": "stage42_aa_retrained_ablation_matrix_pass_with_jepa_transformer_boundary",
            "y": "stage42_y_unified_ablation_evidence_pass",
            "bw": "stage42_bw_safety_floor_necessity_audit_pass",
            "ec": "stage42_ec_group_consistency_contribution_audit_pass",
        },
        "summary": {"supported_or_necessary_modules": 5},
        "module_rows": [
            {"module": "history", "main_claim_allowed": True},
            {"module": "domain_expert", "main_claim_allowed": True},
            {"module": "safe_switch", "main_claim_allowed": True},
            {"module": "teacher_floor", "main_claim_allowed": True},
            {"module": "group_consistency_full_waypoint", "main_claim_allowed": True},
            {"module": "scene_goal", "main_claim_allowed": False},
            {"module": "neighbor_interaction", "main_claim_allowed": False},
            {"module": "JEPA", "main_claim_allowed": False},
            {"module": "Transformer", "main_claim_allowed": False},
        ],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fu._gate(payload)

    assert gate["verdict"] == "stage42_fu_module_contribution_ledger_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_scene_goal_is_overclaimed() -> None:
    payload = {
        "source": fu.SOURCE,
        "input_gates": {
            "aa": "stage42_aa_retrained_ablation_matrix_pass_with_jepa_transformer_boundary",
            "y": "stage42_y_unified_ablation_evidence_pass",
            "bw": "stage42_bw_safety_floor_necessity_audit_pass",
            "ec": "stage42_ec_group_consistency_contribution_audit_pass",
        },
        "summary": {"supported_or_necessary_modules": 5},
        "module_rows": [
            {"module": "history", "main_claim_allowed": True},
            {"module": "domain_expert", "main_claim_allowed": True},
            {"module": "safe_switch", "main_claim_allowed": True},
            {"module": "teacher_floor", "main_claim_allowed": True},
            {"module": "group_consistency_full_waypoint", "main_claim_allowed": True},
            {"module": "scene_goal", "main_claim_allowed": True},
            {"module": "neighbor_interaction", "main_claim_allowed": False},
            {"module": "JEPA", "main_claim_allowed": False},
            {"module": "Transformer", "main_claim_allowed": False},
        ],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fu._gate(payload)

    assert gate["gates"]["scene_goal_overclaim_blocked"] is False
    assert gate["verdict"] == "stage42_fu_module_contribution_ledger_partial"
