from __future__ import annotations

import numpy as np

from src import stage43_bounded_residual_safety_audit as al


def test_clip_residual_by_norm_limits_each_waypoint_norm() -> None:
    residual = np.asarray([[[3.0, 4.0], [0.0, 2.0]]], dtype=np.float32)
    clipped = al._clip_residual_by_norm(residual, 1.0)
    norms = np.linalg.norm(clipped, axis=2)
    assert np.all(norms <= 1.00001)
    assert np.isclose(norms[0, 0], 1.0, atol=1e-5)
    assert np.isclose(norms[0, 1], 1.0, atol=1e-5)


def _payload(*, replay_diff: float = 0.0, safe_improves: bool = True) -> dict:
    stored_all = 0.30
    stored_t50 = 0.16
    stored_hard = 0.28
    safe = {
        "all": 0.32 if safe_improves else 0.29,
        "t50": 0.17 if safe_improves else 0.15,
        "t100": 0.0,
        "hard_failure": 0.30 if safe_improves else 0.27,
        "easy": 0.0,
        "safe_easy": True,
        "safe_t100": True,
    }
    unconstrained = {
        "all": 0.35,
        "t50": 0.20,
        "t100": -0.50,
        "hard_failure": 0.31,
        "easy": 0.20,
        "safe_easy": False,
        "safe_t100": False,
    }
    return {
        "source": al.SOURCE,
        "stored_policy_replay_diff": {"max_abs_diff": replay_diff},
        "feature_schema_match": True,
        "cache_row_hash_match_prior": True,
        "validation_search": {
            "safe_constrained": {"searched_candidates": 10},
            "unconstrained": {"searched_candidates": 10},
        },
        "best_safe_bounded_residual": safe,
        "best_unconstrained_bounded_residual": unconstrained,
        "bounded_residual_deployable_candidate": safe_improves,
        "comparison_to_stored_hard_switch": {
            "safe_minus_stored_all": safe["all"] - stored_all,
            "safe_minus_stored_t50": safe["t50"] - stored_t50,
            "safe_minus_stored_t100": safe["t100"] - (-0.17),
            "safe_minus_stored_hard_failure": safe["hard_failure"] - stored_hard,
            "safe_minus_stored_easy": safe["easy"] - 0.0,
        },
        "interpretation": {
            "global_floor_removable": False,
            "deployment_decision": "promote_bounded_residual_candidate"
            if safe_improves
            else "keep_stage43_m_floor_policy; bounded residual remains diagnostic",
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


def test_gate_promotes_safe_bounded_residual_when_it_beats_reference() -> None:
    gate = al._gate(_payload(safe_improves=True))
    assert gate["passed"] == gate["total"]
    assert gate["deploy_bounded_residual"] is True
    assert gate["verdict"] == "stage43_al_bounded_residual_candidate_pass"


def test_gate_passes_as_honest_diagnostic_when_safe_residual_does_not_beat_reference() -> None:
    gate = al._gate(_payload(safe_improves=False))
    assert gate["passed"] == gate["total"]
    assert gate["deploy_bounded_residual"] is False
    assert gate["verdict"] == "stage43_al_bounded_residual_diagnostic_keep_floor"


def test_gate_fails_when_replay_differs() -> None:
    gate = al._gate(_payload(replay_diff=1e-3, safe_improves=True))
    assert gate["gates"]["stage43_m_exact_replay"] is False
    assert gate["verdict"] == "stage43_al_bounded_residual_audit_incomplete"
