from __future__ import annotations

from src import stage43_integrated_candidate_gate as aq


def _manifest(*, easy: float = 0.0, replay_diff: float = 0.0, complete: bool = False) -> dict:
    return {
        "source": aq.SOURCE,
        "policy_hash": "abc",
        "frozen_replayable_policy_hash": "frozen-abc",
        "input_gate_verdicts": {
            "stage43_aj": "pass",
            "stage43_ak": "pass",
            "stage43_al": "pass",
            "stage43_am": "pass",
            "stage43_an": "pass",
            "stage43_ao": "pass",
            "stage43_ap": "pass",
            "stage43_p": "pass",
        },
        "current_best_deployable": {"global_floor_removed": False, "h100_guarded": True},
        "metrics": {
            "reviewer_replay_max_abs_diff": replay_diff,
            "easy_degradation_vs_floor": easy,
            "t100_raw_frame_diagnostic_vs_floor": 0.0,
            "bootstrap_delta_ci": {
                "all_delta_improvement": {"low": 0.01},
                "t50_delta_improvement": {"low": 0.01},
                "hard_failure_delta_improvement": {"low": 0.01},
            },
            "latest_bootstrap_ci": {
                "full_waypoint_ade_improvement_vs_floor": {"low": 0.10},
                "t50_full_waypoint_ade_improvement_vs_floor": {"low": 0.10},
                "hard_failure_full_waypoint_ade_improvement_vs_floor": {"low": 0.10},
                "easy_degradation_vs_floor": {"high": 0.0},
            },
            "domain_deltas_vs_stored_hard_switch": {"ETH_UCY": 0.01, "TrajNet": 0.02, "UCY": 0.03},
            "horizon_deltas_vs_stored_hard_switch": {"10": 0.01, "25": 0.02, "50": 0.03, "100": 0.04},
            "latest_domain_metrics": {"ETH_UCY": {}, "TrajNet": {}, "UCY": {}},
            "latest_horizon_metrics": {"10": {}, "25": {}, "50": {}, "100": {}},
            "latest_delta_vs_frozen_replay": {"all": 0.10, "t50": 0.10, "hard_failure": 0.10, "easy": 0.0},
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "answers": {
            "is_current_goal_complete": complete,
            "is_a_journal_candidate_now": False,
        },
    }


def test_gate_passes_for_integrated_candidate_without_overclaim() -> None:
    gate = aq._gate(_manifest())
    assert gate["passed"] == gate["total"]
    assert gate["current_candidate_supported"] is True
    assert gate["goal_complete"] is False


def test_gate_fails_when_easy_degradation_is_not_preserved() -> None:
    gate = aq._gate(_manifest(easy=0.05))
    assert gate["gates"]["easy_preserved"] is False
    assert gate["current_candidate_supported"] is False


def test_gate_fails_when_reviewer_replay_is_not_exact() -> None:
    gate = aq._gate(_manifest(replay_diff=0.001))
    assert gate["gates"]["reviewer_replay_exact"] is False
    assert gate["current_candidate_supported"] is False


def test_gate_fails_if_long_goal_is_marked_complete() -> None:
    gate = aq._gate(_manifest(complete=True))
    assert gate["gates"]["long_objective_kept_active"] is False
    assert gate["goal_complete"] is False
