from __future__ import annotations

from src import stage43_paper_evidence_refresh as ap


def _payload(*, overclaim: bool = False, ci_positive: bool = True) -> dict:
    low = 0.01 if ci_positive else -0.01
    return {
        "source": ap.SOURCE,
        "evidence_rows": [
            {"claim_name": "Reviewer-replayable protected bounded residual policy", "status": "supported"},
            {"claim_name": "Latest protected tail-horizon full-waypoint adapter", "status": "supported"},
            {"claim_name": "Protected full-waypoint latent dynamics lift", "status": "supported"},
            {"claim_name": "Bootstrap-supported latest full-test lift", "status": "supported"},
            {"claim_name": "Frozen replay bootstrap-supported delta over stored hard switch", "status": "supported"},
            {"claim_name": "Global floor removal", "status": "supported" if overclaim else "not_supported"},
        ],
        "key_metrics": {
            "ao_replay_diff": 0.0,
            "all": 0.3,
            "t50": 0.2,
            "hard_failure": 0.3,
            "easy": 0.0,
            "latest_ci": {
                "full_waypoint_ade_improvement_vs_floor": {"low": low},
                "t50_full_waypoint_ade_improvement_vs_floor": {"low": low},
                "hard_failure_full_waypoint_ade_improvement_vs_floor": {"low": low},
                "easy_degradation_vs_floor": {"high": 0.0},
            },
            "frozen_replay": {"t50_delta_ci": {"low": low}},
        },
        "answers": {
            "a_journal_candidate": False if not overclaim else True,
            "still_2_5d": True,
            "metric_time_subset_available": False,
            "full_waypoint_dynamics_available": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }


def test_gate_passes_when_evidence_supported_without_overclaim() -> None:
    gate = ap._gate(_payload(overclaim=False, ci_positive=True))
    assert gate["passed"] == gate["total"]
    assert gate["paper_evidence_refreshed"] is True


def test_gate_fails_when_global_floor_removal_is_overclaimed() -> None:
    gate = ap._gate(_payload(overclaim=True, ci_positive=True))
    assert gate["gates"]["global_floor_not_overclaimed"] is False
    assert gate["gates"]["a_journal_not_overclaimed"] is False


def test_gate_fails_when_bootstrap_delta_ci_is_not_positive() -> None:
    gate = ap._gate(_payload(overclaim=False, ci_positive=False))
    assert gate["gates"]["latest_bootstrap_claim_supported"] is False
    assert gate["gates"]["frozen_replay_bootstrap_claim_supported"] is False
    assert gate["paper_evidence_refreshed"] is False
