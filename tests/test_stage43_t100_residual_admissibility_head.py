from __future__ import annotations

import numpy as np

from src import stage43_t100_residual_admissibility_head as ct


def _dummy_ds() -> object:
    ds = type("Dummy", (), {})()
    ds.x = np.zeros((3, 4), dtype=np.float32)
    ds.feature_names = ["history_speed", "floor_endpoint_rel_x", "floor_endpoint_rel_y", "domain_UCY"]
    ds.floor_waypoint_delta = np.zeros((3, 4, 2), dtype=np.float32)
    ds.waypoint_delta = np.zeros((3, 4, 2), dtype=np.float32)
    ds.waypoint_valid = np.ones((3, 4), dtype=bool)
    ds.floor_ade = np.ones(3, dtype=np.float32)
    ds.floor_fde = np.ones(3, dtype=np.float32)
    ds.hard = np.zeros(3, dtype=bool)
    ds.failure = np.zeros(3, dtype=bool)
    ds.easy = np.asarray([True, False, False])
    ds.horizon = np.asarray([100, 100, 100])
    return ds


def _dummy_pred() -> dict[str, np.ndarray]:
    return {
        "residual": np.zeros((3, 4, 2), dtype=np.float32),
        "failure": np.asarray([0.1, 0.8, 0.5], dtype=np.float32),
        "gain": np.asarray([0.2, 0.9, 0.4], dtype=np.float32),
        "harm": np.asarray([0.9, 0.1, 0.2], dtype=np.float32),
        "density": np.asarray([0.1, 0.5, 0.2], dtype=np.float32),
        "latent": np.zeros((3, 8), dtype=np.float32),
    }


def test_augmented_alpha_features_are_train_label_only() -> None:
    ds = _dummy_ds()
    pred = _dummy_pred()
    aug = ct._augment_alpha_features(ds, pred, alphas=np.asarray([0.1, 0.5], dtype=np.float32))
    assert aug["x"].shape[0] == 6
    assert aug["y_gain"].shape == (6,)
    assert "candidate_alpha" in aug["feature_names"].tolist()
    assert not any("future" in str(name).lower() for name in aug["feature_names"].tolist())


def test_policy_metrics_force_easy_floor() -> None:
    ds = _dummy_ds()
    pred = _dummy_pred()
    head = {"gain": np.ones(6, dtype=np.float32), "harm": np.zeros(6, dtype=np.float32), "delta": -np.ones(6, dtype=np.float32)}
    metrics, _ade, _fde, allow = ct._policy_metrics_for_alpha(
        ds,
        pred,
        head,
        alpha_index=0,
        policy={"gain_threshold": 0.0, "harm_threshold": 1.0, "delta_threshold": 1.0, "force_easy_floor": True},
    )
    assert allow[0] == np.bool_(False)
    assert metrics["easy_degradation_vs_floor"] == 0.0


def test_gate_accepts_honest_floor_result() -> None:
    payload = {
        "stage43_cs_precondition": {"verdict": "stage43_cs_t100_bounded_residual_latent_keep_floor"},
        "result_source": "fresh_torch_t100_residual_admissibility_head",
        "checkpoint": "README_RESULTS.md",
        "checkpoint_committed": False,
        "horizon_protocol": {"horizons": [100]},
        "feature_contract": {"denied_feature_name_hits": []},
        "alpha_protocol": {"num_alphas": 7},
        "selection_protocol": {"test_threshold_tuning": False},
        "test_metrics_with_floor": {
            "rows": 10,
            "easy_degradation_vs_floor": 0.0,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.0,
        },
        "deploy_on_current_heldout": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "long_objective_complete": False,
    }
    gate = ct._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_ct_t100_residual_admissibility_keep_floor"
