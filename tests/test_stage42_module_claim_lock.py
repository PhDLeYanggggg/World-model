from __future__ import annotations

from src import stage42_module_claim_lock as gj


def test_summary_locks_supported_and_blocked_modules() -> None:
    inputs = {
        "fu": {
            "summary": {
                "main_claim_allowed_modules": ["history", "domain_expert", "safe_switch", "teacher_floor", "group_consistency_full_waypoint"],
                "blocked_or_auxiliary_modules": ["scene_goal", "neighbor_interaction", "JEPA", "Transformer"],
            }
        },
        "z": {"stage42_z_gate": {"paper_ready_scope": "protected_2p5d", "not_ready_scope": "true3d"}},
        "dp": {"summary": {"closure_decision": "close_current_sequence_graph_residual_context_protocol"}},
        "dq": {
            "summary": {
                "promotion_decision": {
                    "source_level_group_consistency_runtime_policy_promoted": True,
                    "ungated_full_waypoint_deployable": False,
                }
            }
        },
        "gh": {
            "summary": {
                "restricted_metric_time_candidates_after_terms": 5,
                "restricted_ready_now": 0,
                "calibrated_t50_windows_after_terms": 10060,
                "calibrated_t100_windows_after_terms": 5696,
                "domains_with_candidates": ["ETH_UCY", "UCY"],
            }
        },
    }

    summary = gj._summary(inputs)

    assert "history" in summary["supported_main_modules_locked"]
    assert {"scene_goal", "neighbor_interaction", "JEPA", "Transformer"}.issubset(summary["blocked_main_modules_locked"])
    assert summary["protected_full_waypoint_runtime_supported"] is True
    assert summary["ungated_full_waypoint_deployable"] is False
    assert summary["calibrated_subset_ready_now"] == 0
    assert summary["calibrated_t50_after_terms"] == 10060


def test_gate_passes_when_claim_lock_boundaries_hold() -> None:
    payload = {
        "source": gj.SOURCE,
        "input_status": {
            "fu_gate": "stage42_fu_module_contribution_ledger_pass",
            "z_gate": "stage42_z_paper_claim_evidence_audit_pass",
            "dp_gate": "stage42_dp_context_model_closure_pass",
            "dq_gate": "stage42_dq_full_waypoint_promotion_checkpoint_pass",
            "gh_gate": "stage42_gh_calibrated_post_confirmation_subset_plan_pass",
        },
        "summary": {
            "supported_main_modules_locked": [
                "history",
                "domain_expert",
                "safe_switch",
                "teacher_floor",
                "group_consistency_full_waypoint",
            ],
            "blocked_main_modules_locked": ["scene_goal", "neighbor_interaction", "JEPA", "Transformer"],
            "context_protocol_status": "close_current_sequence_graph_residual_context_protocol",
            "protected_full_waypoint_runtime_supported": True,
            "ungated_full_waypoint_deployable": False,
            "calibrated_subset_candidates_after_terms": 5,
            "calibrated_subset_ready_now": 0,
            "next_admissible_experiments": ["a", "b", "c"],
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "post_confirmation_candidates_claimed_as_data": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = gj._gate(payload)

    assert gate["verdict"] == "stage42_gj_module_claim_lock_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_post_confirmation_candidates_are_overclaimed() -> None:
    payload = {
        "source": gj.SOURCE,
        "input_status": {
            "fu_gate": "stage42_fu_module_contribution_ledger_pass",
            "z_gate": "stage42_z_paper_claim_evidence_audit_pass",
            "dp_gate": "stage42_dp_context_model_closure_pass",
            "dq_gate": "stage42_dq_full_waypoint_promotion_checkpoint_pass",
            "gh_gate": "stage42_gh_calibrated_post_confirmation_subset_plan_pass",
        },
        "summary": {
            "supported_main_modules_locked": [
                "history",
                "domain_expert",
                "safe_switch",
                "teacher_floor",
                "group_consistency_full_waypoint",
            ],
            "blocked_main_modules_locked": ["scene_goal", "neighbor_interaction", "JEPA", "Transformer"],
            "context_protocol_status": "close_current_sequence_graph_residual_context_protocol",
            "protected_full_waypoint_runtime_supported": True,
            "ungated_full_waypoint_deployable": False,
            "calibrated_subset_candidates_after_terms": 5,
            "calibrated_subset_ready_now": 0,
            "next_admissible_experiments": ["a", "b", "c"],
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "post_confirmation_candidates_claimed_as_data": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = gj._gate(payload)

    assert gate["gates"]["post_confirmation_candidates_not_overclaimed"] is False
    assert gate["verdict"] == "stage42_gj_module_claim_lock_partial"
