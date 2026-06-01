from __future__ import annotations

from src import stage43_bounded_residual_reviewer_replay as ao


def test_verify_policy_hash_detects_tampering() -> None:
    policy = {"a": 1, "policy_hash": "bad"}
    result = ao._verify_policy_hash(policy)
    assert result["expected"] == "bad"
    assert result["match"] is False


def test_metric_diff_reports_zero_for_matching_metrics() -> None:
    replayed = {
        "full_waypoint_ade_improvement_vs_floor": 0.1,
        "endpoint_fde_improvement_vs_floor": 0.2,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.3,
        "t50_endpoint_fde_improvement_vs_floor": 0.4,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.5,
        "easy_degradation_vs_floor": 0.0,
        "switch_rate": 0.6,
    }
    frozen = {
        "all": 0.1,
        "endpoint": 0.2,
        "t50": 0.3,
        "t50_endpoint": 0.4,
        "t100": 0.0,
        "hard_failure": 0.5,
        "easy": 0.0,
        "switch_rate": 0.6,
    }
    diff = ao._metric_diff(replayed, frozen)
    assert diff["max_abs_diff"] == 0.0


def _payload(*, replay_diff: float = 0.0, hash_match: bool = True) -> dict:
    return {
        "source": ao.SOURCE,
        "policy_artifact": str(ao.FROZEN_POLICY),
        "policy_hash": {"match": hash_match},
        "feature_schema_match": True,
        "cache_row_hash_match_prior": True,
        "checkpoint_hash_match_freeze": True,
        "stage43_m_report_hash_match_freeze": True,
        "checkpoint_not_tracked_by_git": True,
        "replay_diff": {"max_abs_diff": replay_diff},
        "replayed_metrics": {
            "easy_degradation_vs_floor": 0.0,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "thresholds_selected_on_test": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }


def test_gate_passes_for_exact_reviewer_replay() -> None:
    gate = ao._gate(_payload(replay_diff=0.0, hash_match=True))
    assert gate["passed"] == gate["total"]
    assert gate["reviewer_replay_passed"] is True


def test_gate_fails_for_replay_diff() -> None:
    gate = ao._gate(_payload(replay_diff=1e-3, hash_match=True))
    assert gate["gates"]["replay_metrics_exact"] is False
    assert gate["reviewer_replay_passed"] is False


def test_gate_fails_for_policy_hash_mismatch() -> None:
    gate = ao._gate(_payload(replay_diff=0.0, hash_match=False))
    assert gate["gates"]["policy_hash_recomputed"] is False
    assert gate["reviewer_replay_passed"] is False
