from __future__ import annotations

import numpy as np

from src.stage42_constraint_aware_composer import _candidate_score, _composer_key, _deployment_decision, _gate


def _metric(all_imp: float, t50: float, hard: float, easy: float) -> dict:
    return {
        "rows": 100,
        "all_improvement": all_imp,
        "t10_improvement": all_imp,
        "t25_improvement": all_imp,
        "t50_improvement": t50,
        "t100_raw_frame_diagnostic_improvement": 0.1,
        "hard_failure_improvement": hard,
        "easy_degradation": easy,
        "switch_rate": 0.2,
        "harm_over_fallback": -0.1,
    }


def test_stage42_ev_composer_key_includes_domain_horizon_and_risk() -> None:
    data = {
        "dataset": np.asarray(["TrajNet", "UCY", "TrajNet"], dtype=object),
        "horizon": np.asarray([50, 100, 50]),
    }
    ids = np.asarray([0, 1, 2])
    risk = np.asarray([True, False, False])

    keys = _composer_key(data, ids, risk, "domain_horizon_risk")

    assert keys.tolist() == ["TrajNet|50|risk", "UCY|100|clear", "TrajNet|50|clear"]


def test_stage42_ev_candidate_score_penalizes_easy_harm_and_near_collision() -> None:
    safe = _candidate_score(_metric(0.2, 0.2, 0.2, 0.0), {"base_near_005": 0.02, "final_near_005": 0.01})
    unsafe = _candidate_score(_metric(0.2, 0.2, 0.2, 0.08), {"base_near_005": 0.02, "final_near_005": 0.04})

    assert safe > unsafe


def test_stage42_ev_deployment_decision_requires_beating_stage42_di() -> None:
    metric = _metric(0.25, 0.22, 0.24, -0.1)
    promotes = _deployment_decision(
        metric,
        {
            "delta_vs_stage42_di": {"all_improvement": 0.01, "hard_failure_improvement": 0.01},
            "delta_vs_stage42_eu": {"all_improvement": 0.02},
        },
    )
    diagnostic = _deployment_decision(
        metric,
        {
            "delta_vs_stage42_di": {"all_improvement": -0.01, "hard_failure_improvement": 0.01},
            "delta_vs_stage42_eu": {"all_improvement": 0.02},
        },
    )

    assert promotes["promote_constraint_aware_composer"] is True
    assert diagnostic["promote_constraint_aware_composer"] is False
    assert diagnostic["diagnostic_positive"] is True


def test_stage42_ev_gate_passes_for_promotable_composer() -> None:
    payload = {
        "candidate_families": ["floor", "stage42_am", "stage42_di", "stage42_eu"],
        "composer": {
            "mode_rows": [{}, {}, {}],
            "selected": {
                "mode": "domain_horizon_risk",
                "test_metric_vs_floor": _metric(0.25, 0.22, 0.24, -0.1),
                "test_diagnostics": {"base_near_005": 0.02, "final_near_005": 0.01},
            },
        },
        "comparison_to_prior": {
            "delta_vs_stage42_di": {"all_improvement": 0.01, "hard_failure_improvement": 0.01},
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_composer_selection": True,
            "train_only_feature_normalization": True,
            "source_overlap_pass": True,
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
    assert gate["verdict"] == "stage42_ev_constraint_aware_composer_pass_promotable"
