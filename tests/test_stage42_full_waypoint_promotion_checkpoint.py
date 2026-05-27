from __future__ import annotations

from src import stage42_full_waypoint_promotion_checkpoint as dq


def _payload() -> dict:
    summary = {
        "full_waypoint_transformer_protected_vs_floor": {"all_improvement": 0.1, "t50_improvement": 0.1, "hard_failure_improvement": 0.1, "easy_degradation": 0.0},
        "ungated_full_waypoint_transformer_vs_floor": {"all_improvement": 0.3, "t50_improvement": 0.2, "hard_failure_improvement": 0.3, "easy_degradation": 1.0},
        "group_consistency_repair_vs_train_horizon_causal_floor": {"all_improvement": 0.2, "t50_improvement": 0.1, "hard_failure_improvement": 0.2, "easy_degradation": 0.0},
        "group_consistency_delta_vs_stage42_am": {"all_improvement": 0.01, "hard_failure_improvement": 0.01},
        "group_consistency_safety": {
            "runtime_base_near_005": 0.02,
            "runtime_final_near_005": 0.01,
            "switch_exact_match": True,
            "selected_xy_max_abs_diff": 0.0,
            "selected_ade_max_abs_diff": 0.0,
            "selected_fde_max_abs_diff": 0.0,
        },
        "context_closure_decision": "close_current_sequence_graph_residual_context_protocol",
        "promotion_decision": {"global_primary_full_waypoint_replacement_claim_allowed": False},
    }
    return {
        "source": "fresh_synthesis_after_da3_full_waypoint_rerun",
        "inputs": {
            "stage42_c_full_waypoint_dynamics": {"source": "fresh_run"},
            "stage42_co_common_validation_composer": {"source": "fresh_common_validation_eval_from_cached_verified_checkpoints"},
            "stage42_di_group_consistency_repair": {"source": "fresh_stage42_di_group_consistency_full_waypoint_repair"},
            "stage42_dl_runtime_replay": {"source": "fresh_runtime_api_from_frozen_group_consistency_policy_artifact"},
        },
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_gate_passes_for_exact_replay_protected_full_waypoint() -> None:
    gate = dq._gate(_payload())
    assert gate["verdict"] == "stage42_dq_full_waypoint_promotion_checkpoint_pass"
    assert gate["passed"] == gate["total"]


def test_gate_blocks_ungated_full_waypoint_if_not_marked_unsafe() -> None:
    payload = _payload()
    payload["summary"]["ungated_full_waypoint_transformer_vs_floor"]["easy_degradation"] = 0.0
    gate = dq._gate(payload)
    assert gate["gates"]["ungated_full_waypoint_blocked"] is False
    assert gate["passed"] < gate["total"]
