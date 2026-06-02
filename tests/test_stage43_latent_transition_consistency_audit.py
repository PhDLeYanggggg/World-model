from __future__ import annotations

import numpy as np

from src import stage43_latent_transition_consistency_audit as by


def test_transition_metrics_rewards_dynamics_over_identity_and_centroid() -> None:
    z_target = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    z_t = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    z_next = np.asarray([[0.9, 0.0], [2.1, 0.0], [3.0, 0.0]], dtype=np.float32)
    centroid = np.asarray([1.0, 0.0], dtype=np.float32)
    out = by._transition_metrics(z_t, z_next, z_target, centroid, np.ones(3, dtype=bool))
    assert out["rows"] == 3
    assert out["transition_gain_vs_identity"] > 0.0
    assert out["transition_gain_vs_train_centroid"] > 0.0
    assert out["mean_cosine_next_target"] >= out["mean_cosine_identity_target"]


def test_weak_slices_flags_negative_transition_gain() -> None:
    table = {
        "10": {"rows": 10, "transition_gain_vs_identity": 0.1, "transition_gain_vs_train_centroid": 0.2, "mean_cosine_next_target": 0.8},
        "100": {"rows": 10, "transition_gain_vs_identity": -0.1, "transition_gain_vs_train_centroid": 0.2, "mean_cosine_next_target": 0.4},
    }
    weak = by._weak_slices(table, axis="horizon")
    assert len(weak) == 1
    assert weak[0]["slice"] == "100"


def test_gate_allows_caveated_positive_latent_transition() -> None:
    payload = {
        "source": by.SOURCE,
        "stage43_m_precondition": {"checkpoint_sha256_matches_stage43_m": True},
        "stage43_bx_precondition": {"verdict": "stage43_bx_latent_risk_head_robustness_pass_horizon_caveat"},
        "evaluation_protocol": {"test_rows": 100, "future_target_latent_label_eval_only": True},
        "overall": {
            "transition_gain_vs_identity": 0.2,
            "transition_gain_vs_train_centroid": -0.1,
        },
        "calibrated_readout_overall": {
            "transition_gain_vs_identity": -0.01,
            "transition_gain_vs_train_centroid": 0.3,
        },
        "bootstrap": {
            "transition_gain_vs_identity": {"low": 0.1},
            "transition_gain_vs_train_centroid": {"low": -0.2},
        },
        "calibrated_readout_bootstrap": {
            "transition_gain_vs_identity": {"low": -0.02},
            "transition_gain_vs_train_centroid": {"low": 0.1},
        },
        "latent_stats": {
            "z_next_min_variance": 0.02,
            "target_min_variance": 0.02,
            "noncollapse_threshold": 0.01,
        },
        "by_domain": {
            "UCY": {"transition_gain_vs_identity": 0.1, "transition_gain_vs_train_centroid": 0.1},
            "ETH_UCY": {"transition_gain_vs_identity": 0.1, "transition_gain_vs_train_centroid": 0.1},
        },
        "by_horizon": {
            "10": {"transition_gain_vs_identity": 0.1, "transition_gain_vs_train_centroid": 0.1},
            "25": {"transition_gain_vs_identity": 0.1, "transition_gain_vs_train_centroid": 0.1},
            "50": {"transition_gain_vs_identity": 0.1, "transition_gain_vs_train_centroid": 0.1},
            "100": {"transition_gain_vs_identity": -0.01, "transition_gain_vs_train_centroid": 0.1},
        },
        "weak_transition_slices": [{"axis": "horizon", "slice": "100"}],
        "calibrated_readout_weak_transition_slices": [{"axis": "horizon", "slice": "100"}],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_target_latent_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = by._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_by_latent_transition_consistency_pass_with_readout_caveat"
