from __future__ import annotations

from src.stage42_post_eq_context_claim_refresh import (
    _context_claim_decision,
    _context_evidence_matrix,
    _gate,
    _updated_actions,
)


def _eq() -> dict:
    return {
        "source": "fresh_stage42_sequence_graph_context_router",
        "stage42_eq_gate": {"verdict": "stage42_eq_sequence_graph_context_router_pass"},
        "best_router": "baseline_plus_history_goal_neighbor",
        "positive_sequence_graph_context_routers": [],
        "summary": {
            "sequence_graph_increment_verdict": "stage42_eq_sequence_graph_context_router_not_supported",
            "best_router_test_metric_vs_baseline_family": {
                "all_improvement": 0.000118,
                "t50_improvement": -0.000197,
                "t100_raw_frame_diagnostic_improvement": 0.000083,
                "hard_failure_improvement": 0.000169,
                "easy_degradation": -0.001971,
            },
        },
    }


def _el() -> dict:
    return {
        "positive_context_gain_routers": [],
        "best_router": "baseline_plus_history_goal_neighbor",
        "summary": {
            "context_increment_verdict": "stage42_el_context_gain_router_not_supported",
            "best_router_test_metric_vs_baseline_family": {
                "all_improvement": 0.000278,
                "t50_improvement": -0.000019,
                "hard_failure_improvement": 0.000321,
            },
        },
    }


def _ar() -> dict:
    return {
        "positive_sequence_context_variants": [],
        "summary": {"best_variant": "sequence_history", "sequence_context_verdict": "stage42_ar_sequence_context_not_supported"},
        "sequence_deltas": {
            "sequence_history": {
                "delta_vs_baseline_family_only": {
                    "all_improvement": -0.024,
                    "t50_improvement": -0.083,
                    "hard_failure_improvement": -0.028,
                }
            }
        },
    }


def _as() -> dict:
    return {
        "positive_graph_context_variants": [],
        "summary": {"graph_context_verdict": "stage42_as_graph_context_not_supported"},
        "graph_deltas": {
            "graph_history_goal": {
                "delta_vs_baseline_family_only": {
                    "all_improvement": -0.023,
                    "t50_improvement": -0.086,
                    "hard_failure_improvement": -0.026,
                }
            }
        },
    }


def _da() -> dict:
    return {
        "next_actions": [
            {"id": "DA-1", "priority": 100, "status": "not_run_next_action", "title": "source", "requires_user_or_external_state": True},
            {"id": "DA-2", "priority": 92, "status": "not_run_next_action", "title": "context", "requires_user_or_external_state": False},
        ]
    }


def _em() -> dict:
    return {"summary": {"conversion_ready_now": 0, "auto_download_allowed_now": 0, "manual_terms_required_targets": 5}}


def _en() -> dict:
    return {
        "summary": {
            "floor_free_neural_deployable": False,
            "global_floor_removal_allowed": False,
            "partial_relaxation_components": ["t50_slice_relaxation::UCY|50"],
        }
    }


def test_stage42_er_context_decision_closes_negative_protocols() -> None:
    matrix = _context_evidence_matrix(_eq(), _el(), _ar(), _as())
    decision = _context_claim_decision(matrix)

    assert len(matrix) == 4
    assert decision["independent_context_main_claim_allowed"] is False
    assert decision["decision"] == "close_current_shallow_sequence_graph_context_protocol"
    assert "sequence_graph_context_router" in decision["closed_protocols"]


def test_stage42_er_updated_actions_close_da2_and_prioritize_source() -> None:
    matrix = _context_evidence_matrix(_eq(), _el(), _ar(), _as())
    decision = _context_claim_decision(matrix)
    actions = _updated_actions(_da(), decision, _em(), _en())

    assert actions[0]["id"] == "ER-1"
    assert actions[0]["requires_user_or_external_state"] is True
    assert any(row["id"] == "DA-2" and row["status"] == "closed_negative_fresh_run" for row in actions)


def test_stage42_er_gate_passes_for_bounded_post_eq_refresh() -> None:
    matrix = _context_evidence_matrix(_eq(), _el(), _ar(), _as())
    decision = _context_claim_decision(matrix)
    payload = {
        "inputs": {"stage42_eq": {"verdict": "stage42_eq_sequence_graph_context_router_pass"}},
        "paper_refresh_summary": {
            "context_evidence_matrix": matrix,
            "context_claim_decision": decision,
            "updated_next_actions": _updated_actions(_da(), decision, _em(), _en()),
            "paper_verdict": {
                "source_legal_blocker_still_primary": True,
                "floor_free_neural_deployable": False,
                "global_floor_removal_allowed": False,
            },
        },
        "paper_file_status": [
            {"updated": True, "contains_stage42_er": True, "contains_boundaries": True},
            {"updated": True, "contains_stage42_er": True, "contains_boundaries": True},
        ],
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_er_post_eq_context_claim_refresh_pass"
