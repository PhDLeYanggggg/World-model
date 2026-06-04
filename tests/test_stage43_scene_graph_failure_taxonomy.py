from __future__ import annotations

from src import stage43_scene_graph_failure_taxonomy as dk


def _payload(
    *,
    fusion_positive: bool = False,
    raw_scene_overclaim: bool = False,
    graph_not_safe: bool = False,
) -> dict:
    return {
        "source": dk.SOURCE,
        "input_verdicts": {
            "stage43_bq": "stage43_bq_gated_scene_graph_fusion_pass_safe_no_lift_diagnostic",
        },
        "preconditions": {
            "stage43_bl": True,
            "stage43_ag": True,
            "stage43_bo": True,
            "stage43_bp": True,
            "stage43_bq": True,
            "stage43_dj": True,
        },
        "signals": {
            "graph": {
                "no_graph": {"t50": 0.10, "hard_failure": 0.10, "easy_degradation": 0.10},
                "full_graph": {
                    "t50": 0.30,
                    "hard_failure": 0.20,
                    "easy_degradation": 0.03 if graph_not_safe else 0.0,
                },
            },
            "scene_proxy": {
                "no_scene": {"t50": 0.20, "easy_degradation": 0.08},
                "geometry_route": {"t50": 0.25, "easy_degradation": 0.0},
                "full_scene": {"easy_degradation": 0.09},
            },
            "fusion": {
                "scene_graph_full": {
                    "t50": 0.05 if not fusion_positive else 0.40,
                    "easy_degradation": 0.13,
                },
                "graph_history_only": {"t50": 0.16},
                "deltas": {
                    "gated_fusion_minus_best_single": {
                        "t50": -0.14 if not fusion_positive else 0.05,
                    }
                },
                "bq_t50_contribution_ci": {
                    "high": -0.29 if not fusion_positive else 0.02,
                },
            },
        },
        "next_training_contract": {
            "train_next": True,
            "name": "stage43_next_graph_first_scene_residual_moe",
        },
        "decision": {
            "deployable_policy_changed": False,
            "raw_scene_sdf_still_blocked": not raw_scene_overclaim,
            "t100_raw_frame_still_diagnostic": True,
        },
        "no_leakage": {
            "bp": {
                "future_endpoint_input": False,
                "future_waypoint_input": False,
                "central_velocity_input": False,
                "test_endpoint_goal_construction": False,
                "test_statistics_normalization": False,
            },
            "bq": {
                "future_endpoint_input": False,
                "future_waypoint_input": False,
                "central_velocity_input": False,
                "test_endpoint_goal_construction": False,
                "test_statistics_normalization": False,
            },
            "dj": {
                "future_endpoint_input": False,
                "future_waypoint_input": False,
                "central_velocity_input": False,
                "test_endpoint_goal_construction": False,
                "test_statistics_normalization": False,
            },
        },
        "claim_boundary": {
            "bl": {
                "true_3d_world_model": False,
                "foundation_world_model": False,
                "metric_or_seconds_claim": False,
                "stage5c_executed": False,
                "smc_enabled": False,
            },
            "bp": {
                "true_3d_world_model": False,
                "foundation_world_model": False,
                "metric_or_seconds_claim": False,
                "stage5c_executed": False,
                "smc_enabled": False,
            },
            "bq": {
                "true_3d_world_model": False,
                "foundation_world_model": False,
                "metric_or_seconds_claim": False,
                "stage5c_executed": False,
                "smc_enabled": False,
            },
            "current": {
                "true_3d_world_model": False,
                "foundation_world_model": False,
                "metric_or_seconds_claim": False,
                "dataset_local_raw_frame_only": True,
                "raw_scene_or_verified_sdf_claim": raw_scene_overclaim,
                "stage5c_executed": False,
                "smc_enabled": False,
                "long_objective_complete": False,
            },
        },
    }


def test_gate_passes_when_scene_graph_failure_is_honestly_identified() -> None:
    gate = dk._gate(_payload())
    assert gate["verdict"] == "stage43_dk_scene_graph_failure_taxonomy_pass_next_graph_first_moe"
    assert gate["deployable_policy_changed"] is False
    assert gate["next_training_contract"] == "stage43_next_graph_first_scene_residual_moe"


def test_gate_fails_if_gated_fusion_is_not_actually_negative() -> None:
    gate = dk._gate(_payload(fusion_positive=True))
    assert gate["gates"]["gated_fusion_safe_no_lift_identified"] is False
    assert gate["gates"]["bootstrap_confirms_negative_gated_t50"] is False


def test_gate_fails_if_raw_scene_blocker_is_overclaimed() -> None:
    gate = dk._gate(_payload(raw_scene_overclaim=True))
    assert gate["gates"]["raw_scene_sdf_not_overclaimed"] is False
    assert gate["gates"]["claim_boundary_not_overstated"] is False


def test_gate_fails_if_graph_signal_hurts_easy_cases() -> None:
    gate = dk._gate(_payload(graph_not_safe=True))
    assert gate["gates"]["graph_signal_positive_and_easy_safe"] is False
