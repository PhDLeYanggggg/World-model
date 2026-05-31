from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from src import stage43_auxiliary_head_repair as w


def test_ridge_fit_recovers_linear_signal() -> None:
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    weight = w._ridge_fit(x, y, 0.0)
    pred = w._ridge_predict(x, weight)
    assert np.allclose(pred, y, atol=1e-5)


def test_standardize_features_uses_train_statistics() -> None:
    train = np.asarray([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32)
    val = np.asarray([[5.0, 2.0]], dtype=np.float32)
    test = np.asarray([[7.0, 2.0]], dtype=np.float32)
    train_z, val_z, test_z, mean, std = w._standardize_features(train, val, test)
    assert np.allclose(mean, [2.0, 2.0])
    assert np.allclose(std, [1.0, 1.0])
    assert np.allclose(train_z[:, 1], [0.0, 0.0])
    assert np.allclose(val_z, [[3.0, 0.0]])
    assert np.allclose(test_z, [[5.0, 0.0]])


def test_gate_requires_density_repair_positive() -> None:
    payload = {
        "source": w.SOURCE,
        "stage43_v_precondition": {"verdict": "stage43_v_world_state_head_audit_partial"},
        "training_protocol": {
            "checkpoint_sha256_matches_stage43_m": True,
            "selection_data": "train_val_selected_test_once",
            "test_threshold_tuning": False,
        },
        "density_repair": {
            "selected": {
                "delta_r2_vs_stage43m": 0.4,
                "delta_rmse_vs_stage43m": 0.1,
                "test_metrics": {"r2": 0.2},
            }
        },
        "waypoint_validity_proxy_repair": {"selected": {"test_metrics": {"rows": 10}}},
        "claim_boundary": {
            "physical_validity_true_claim": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "no_leakage": {
            "test_threshold_tuning": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
        },
    }
    gate = w._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["deploy_density_proxy_head"] is True
    assert gate["deploy_true_physical_validity"] is False


def test_causal_x_feature_set_uses_dataset_feature_names() -> None:
    ds = SimpleNamespace(
        x=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        feature_names=["speed", "history_density"],
        horizon=np.asarray([50, 100]),
        source_file=np.asarray(["a", "bb"]),
        domain=np.asarray(["UCY", "TrajNet"]),
    )
    pred = {
        "latent": np.zeros((2, 3), dtype=np.float32),
        "failure": np.zeros(2, dtype=np.float32),
        "gain": np.zeros(2, dtype=np.float32),
        "harm": np.zeros(2, dtype=np.float32),
        "density": np.zeros(2, dtype=np.float32),
    }
    x, names = w._calibrator_features(ds, pred, "causal_x")
    assert np.allclose(x, ds.x)
    assert names == ds.feature_names


def test_gate_rejects_negative_density_r2() -> None:
    payload = {
        "source": w.SOURCE,
        "stage43_v_precondition": {"verdict": "stage43_v_world_state_head_audit_partial"},
        "training_protocol": {
            "checkpoint_sha256_matches_stage43_m": True,
            "selection_data": "train_val_selected_test_once",
            "test_threshold_tuning": False,
        },
        "density_repair": {
            "selected": {
                "delta_r2_vs_stage43m": 0.4,
                "delta_rmse_vs_stage43m": 0.1,
                "test_metrics": {"r2": -0.1},
            }
        },
        "waypoint_validity_proxy_repair": {"selected": {"test_metrics": {"rows": 10}}},
        "claim_boundary": {
            "physical_validity_true_claim": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "no_leakage": {
            "test_threshold_tuning": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
        },
    }
    gate = w._gate(payload)
    assert gate["passed"] < gate["total"]
    assert gate["deploy_density_proxy_head"] is False
