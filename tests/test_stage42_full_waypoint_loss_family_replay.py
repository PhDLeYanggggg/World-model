from __future__ import annotations

from src.stage42_full_waypoint_loss_family_replay import _gate, summarize_candidate


def _fake_result(all_delta: float, hard_delta: float) -> dict:
    return {
        "source": "fresh_stage42_dg_full_waypoint_all_hard_loss_repair",
        "stage42_dg_gate": {"verdict": "ok", "passed": 1, "total": 1},
        "model": {
            "selected": {"variant": "balanced", "lambda": 1.0, "val_score": 1.0},
            "metrics": {
                "protected_selected_loss_variant": {
                    "all_improvement": 0.2,
                    "t50_improvement": 0.1,
                    "t100_raw_frame_diagnostic_improvement": 0.05,
                    "hard_failure_improvement": 0.2,
                    "easy_degradation": 0.0,
                }
            },
        },
        "comparison_to_stage42_am": {
            "delta_vs_stage42_am": {
                "all_improvement": all_delta,
                "t50_improvement": 0.0,
                "t100_raw_frame_diagnostic_improvement": 0.0,
                "hard_failure_improvement": hard_delta,
                "easy_degradation": 0.0,
            }
        },
        "deployment_decision": {"decision": "test"},
    }


def test_summarize_candidate_requires_beating_stage42_am_all_and_hard() -> None:
    promoted = summarize_candidate("candidate", _fake_result(0.01, 0.02))
    blocked = summarize_candidate("candidate", _fake_result(0.01, -0.02))

    assert promoted["promotable_over_stage42_am"] is True
    assert blocked["promotable_over_stage42_am"] is False


def test_gate_passes_blocker_when_no_candidate_promotable() -> None:
    payload = {
        "dg_result": {"source": "fresh_stage42_dg_full_waypoint_all_hard_loss_repair"},
        "dh_result": {"source": "fresh_stage42_dh_full_waypoint_proximity_occupancy_loss_repair"},
        "candidate_summaries": [{"selected_val_score": 1.0}, {"selected_val_score": 2.0}],
        "summary": {
            "promotion_decision": "do_not_promote_keep_stage42_am_or_cq_floor",
            "any_promotable_over_stage42_am": False,
            "promotion_blockers": ["no_loss_family_candidate_beats_stage42_am_on_all_and_hard"],
        },
        "no_leakage": {
            "future_waypoint_input": False,
            "future_endpoint_input": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_dx_loss_family_replay_pass_blocker_confirmed"
