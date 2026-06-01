from __future__ import annotations

from src import stage43_current_candidate_reconciliation as ay


def _payload(*, uniform_overclaim: bool = False, source_negative: bool = False, easy: float = 0.0) -> dict:
    source_count = 2 if source_negative else 3
    return {
        "source": ay.SOURCE,
        "input_gate_verdicts": {
            "stage43_p": "pass",
            "stage43_ap": "pass",
            "stage43_ao": "pass",
            "stage43_ax": "pass",
            "stage43_aq": "pass",
        },
        "roles": {
            "performance_leader": {
                "policy_hash": "p" * 64,
                "role": "performance",
                "metrics": {
                    "all": 0.50,
                    "t50": 0.51,
                    "t100_raw_frame_diagnostic": 0.0,
                    "hard_failure": 0.48,
                    "easy_degradation": easy,
                },
                "bootstrap_ci": {
                    "all": {"low": 0.49},
                    "t50": {"low": 0.50},
                    "hard_failure": {"low": 0.47},
                },
                "source_status": {"uniform_positive_transfer": uniform_overclaim},
            },
            "source_horizon_replay_leader": {
                "policy_hash": "x" * 64,
                "role": "source_replay",
                "metrics": {
                    "all": 0.23,
                    "t50": 0.13,
                    "hard_failure": 0.25,
                    "easy_degradation": 0.0,
                },
                "bootstrap_ci": {
                    "all": {"ci_low": 0.22},
                    "t50": {"ci_low": 0.12},
                    "hard_failure": {"ci_low": 0.24},
                    "easy_degradation": {"ci_high": 0.0},
                },
                "source_status": {"domains": ["ETH_UCY", "TrajNet", "UCY"], "nonnegative_all_domains": source_count},
            },
            "frozen_reviewer_replay_artifact": {
                "policy_hash": "a" * 64,
                "replay_diff": 0.0,
                "metrics": {
                    "all": 0.38,
                    "t50": 0.27,
                    "hard_failure": 0.38,
                    "easy_degradation": 0.0,
                },
            },
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "uniform_positive_external_transfer_claim": uniform_overclaim,
            "long_objective_complete": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
    }


def test_gate_passes_when_candidate_roles_are_separated() -> None:
    gate = ay._gate(_payload())
    assert gate["passed"] == gate["total"]
    assert gate["current_candidate_supported"] is True
    assert gate["goal_complete"] is False


def test_gate_fails_if_uniform_transfer_is_overclaimed() -> None:
    gate = ay._gate(_payload(uniform_overclaim=True))
    assert gate["gates"]["uniform_positive_transfer_not_overclaimed"] is False
    assert gate["current_candidate_supported"] is False


def test_gate_fails_if_source_safe_replay_has_negative_source() -> None:
    gate = ay._gate(_payload(source_negative=True))
    assert gate["gates"]["source_safe_candidate_has_all_sources_nonnegative"] is False
    assert gate["current_candidate_supported"] is False


def test_gate_fails_if_performance_leader_hurts_easy_cases() -> None:
    gate = ay._gate(_payload(easy=0.03))
    assert gate["gates"]["performance_leader_supported"] is False
    assert gate["current_candidate_supported"] is False
