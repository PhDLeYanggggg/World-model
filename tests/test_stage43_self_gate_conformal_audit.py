from __future__ import annotations

from src import stage43_self_gate_conformal_audit as ak


def _payload(*, replay_diff: float = 0.0, unsafe_ungated: bool = True) -> dict:
    stored_t100 = -0.17
    return {
        "source": ak.SOURCE,
        "stage43_m_source": {"checkpoint": "outputs/stage43_latent_state/checkpoints/stage43_full_waypoint_latent_dynamics.pt"},
        "feature_schema_match": True,
        "cache_row_hash_match_prior": True,
        "stored_policy_replay_diff": {"max_abs_diff": replay_diff},
        "policy_table": [
            {
                "name": "ungated_neural",
                "easy_degradation_vs_floor": 0.55 if unsafe_ungated else 0.0,
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": -0.72 if unsafe_ungated else stored_t100,
                "safe_easy": not unsafe_ungated,
            },
            {
                "name": "stored_stage43_m_self_gate",
                "easy_degradation_vs_floor": 0.0,
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": stored_t100,
                "safe_easy": True,
            },
            {
                "name": "fresh_self_gate_search",
                "easy_degradation_vs_floor": 0.0,
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": stored_t100,
                "safe_easy": True,
            },
            {
                "name": "conformal_style_h100_easy_guard",
                "easy_degradation_vs_floor": 0.0,
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
                "safe_easy": True,
            },
        ],
        "interpretation": {"global_floor_removable": False},
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


def test_gate_passes_when_replay_exact_and_ungated_unsafe_is_reported() -> None:
    gate = ak._gate(_payload(replay_diff=0.0, unsafe_ungated=True))
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_ak_self_gate_conformal_audit_pass"
    assert gate["global_floor_removable"] is False


def test_gate_fails_when_stored_policy_replay_differs() -> None:
    gate = ak._gate(_payload(replay_diff=1e-3, unsafe_ungated=True))
    assert gate["gates"]["stored_policy_exact_replay"] is False
    assert gate["verdict"] == "stage43_ak_self_gate_conformal_audit_incomplete"


def test_gate_fails_when_ungated_risk_is_not_identified() -> None:
    gate = ak._gate(_payload(replay_diff=0.0, unsafe_ungated=False))
    assert gate["gates"]["ungated_neural_reported_unsafe"] is False
    assert gate["verdict"] == "stage43_ak_self_gate_conformal_audit_incomplete"
