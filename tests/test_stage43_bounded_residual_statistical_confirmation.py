from __future__ import annotations

import numpy as np

from src import stage43_bounded_residual_statistical_confirmation as am


def test_easy_degradation_delta_uses_degradation_not_improvement() -> None:
    floor = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    stored = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    bounded = np.asarray([1.0, 1.1, 1.0], dtype=np.float32)
    ids = np.asarray([0, 1, 2])
    assert am._easy_degradation(stored, floor, ids) == 0.0
    assert am._easy_degradation(bounded, floor, ids) > 0.0


def _payload(*, ci_positive: bool = True, easy_safe: bool = True) -> dict:
    low = 0.01 if ci_positive else -0.01
    easy_high = 0.0 if easy_safe else 0.05
    return {
        "source": am.SOURCE,
        "stage43_al_source": {"deploy_bounded_residual": True},
        "stored_policy_replay_diff": {"max_abs_diff": 0.0},
        "feature_schema_match": True,
        "cache_row_hash_match_prior": True,
        "bootstrap_delta_ci": {
            "n": 2000,
            "metrics": {
                "all_delta_improvement": {"low": low},
                "t50_delta_improvement": {"low": low},
                "hard_failure_delta_improvement": {"low": low},
                "t100_delta_improvement": {"low": 0.0},
                "easy_degradation_bounded": {"high": easy_high},
            },
        },
        "bounded_metrics": {"easy_degradation_vs_floor": 0.0 if easy_safe else 0.04},
        "slice_summary": {"domain_count": 3},
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


def test_gate_confirms_when_delta_ci_positive_and_easy_safe() -> None:
    gate = am._gate(_payload(ci_positive=True, easy_safe=True))
    assert gate["passed"] == gate["total"]
    assert gate["bounded_residual_statistically_confirmed"] is True


def test_gate_fails_when_t50_delta_ci_crosses_zero() -> None:
    gate = am._gate(_payload(ci_positive=False, easy_safe=True))
    assert gate["gates"]["t50_delta_ci_positive"] is False
    assert gate["bounded_residual_statistically_confirmed"] is False


def test_gate_fails_when_easy_ci_is_unsafe() -> None:
    gate = am._gate(_payload(ci_positive=True, easy_safe=False))
    assert gate["gates"]["easy_degradation_ci_safe"] is False
    assert gate["bounded_residual_statistically_confirmed"] is False
