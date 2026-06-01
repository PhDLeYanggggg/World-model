from __future__ import annotations

from src import stage43_source_horizon_expert_replay as ax


def test_metric_diff_exact_for_matching_aw_metrics() -> None:
    metrics = {
        "all_improvement_vs_floor": 0.1,
        "t50_improvement_vs_floor": 0.2,
        "t100_raw_frame_diagnostic_vs_floor": 0.03,
        "hard_failure_improvement_vs_floor": 0.4,
        "easy_degradation_vs_floor": 0.0,
        "switch_rate": 0.5,
    }
    diff = ax._metric_diff(metrics, metrics)
    assert diff["max_abs_diff"] == 0.0


def _payload(*, diff: float = 0.0, leak: bool = False) -> dict:
    return {
        "artifact": str(ax.AW_JSON),
        "artifact_deployment_decision": "candidate_requires_reviewer_replay_before_deployment",
        "policy_hash": "a" * 64,
        "row_hash": "b" * 64,
        "switch_hash": "c" * 64,
        "metric_diff": {"max_abs_diff": diff},
        "replay_metrics": {
            "all_improvement_vs_floor": 0.1,
            "easy_degradation_vs_floor": 0.0,
        },
        "bootstrap": {"unit_t50": {"ci_low": 0.01}},
        "replay_flags": {"domain_easy_safe": True, "negative_source_count": 0},
        "no_leakage": {
            "future_endpoint_input": leak,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "thresholds_selected_on_test": False,
            "replay_does_not_reselect_policy": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }


def test_gate_passes_for_exact_replay_payload() -> None:
    gate = ax._gate(_payload())
    assert gate["passed"] == gate["total"]
    assert gate["reviewer_replay_passed"] is True
    assert gate["candidate_for_deployment_update"] is True


def test_gate_fails_for_metric_diff() -> None:
    gate = ax._gate(_payload(diff=1e-3))
    assert gate["gates"]["replay_metrics_exact"] is False
    assert gate["reviewer_replay_passed"] is False


def test_gate_fails_for_future_leakage() -> None:
    gate = ax._gate(_payload(leak=True))
    assert gate["gates"]["no_future_or_test_leakage"] is False
    assert gate["reviewer_replay_passed"] is False
