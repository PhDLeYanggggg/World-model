from __future__ import annotations

import numpy as np

from src import stage43_interaction_validity_proxy as x


def _toy_cache() -> dict[str, np.ndarray]:
    return {
        "source_file": np.asarray(["s", "s", "s"]),
        "frame_id": np.asarray([1.0, 1.0, 1.0]),
        "horizon": np.asarray([50, 50, 50]),
        "agent_id": np.asarray([1, 2, 3]),
        "current_xy": np.asarray([[0.0, 0.0], [10.0, 0.0], [100.0, 0.0]], dtype=np.float32),
        "waypoint_xy": np.asarray(
            [
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
                [[1.1, 0.0], [2.2, 0.0], [3.3, 0.0], [4.4, 0.0]],
                [[90.0, 0.0], [91.0, 0.0], [92.0, 0.0], [93.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        "waypoint_valid": np.ones((3, 4), dtype=bool),
        "scale": np.ones(3, dtype=np.float32),
    }


def test_future_min_neighbor_distance_groups_by_frame_and_horizon() -> None:
    result = x._future_min_neighbor_distance(_toy_cache())
    dist = result["min_future_neighbor_distance"]
    assert result["grouped_rows"].tolist() == [True, True, True]
    assert float(dist[0]) < 0.11
    assert float(dist[1]) < 0.11
    assert float(dist[2]) > 80.0


def test_smoothness_proxy_stays_in_unit_interval() -> None:
    proxy = x._smoothness_proxy(_toy_cache())
    assert proxy.shape == (3,)
    assert np.all(proxy >= 0.0)
    assert np.all(proxy <= 1.0)
    assert float(proxy[0]) > 0.0


def test_gate_accepts_proxy_signal_without_true_physical_claim() -> None:
    payload = {
        "source": x.SOURCE,
        "stage43_w_precondition": {"verdict": "stage43_w_density_proxy_repaired_validity_proxy_diagnostic"},
        "training_protocol": {
            "checkpoint_sha256_matches_stage43_m": True,
            "selection_data": "train_val_selected_test_once",
            "test_threshold_tuning": False,
        },
        "interaction_risk_head": {
            "selected": {
                "test_metrics": {
                    "defined": True,
                    "auroc": 0.7,
                    "auprc": 0.5,
                    "positive_rate": 0.1,
                    "ece": 0.05,
                }
            }
        },
        "smoothness_validity_proxy_head": {
            "selected": {"test_metrics": {"rows": 100, "r2": 0.1, "corr": 0.4}}
        },
        "no_leakage": {
            "future_waypoint_input": False,
            "future_waypoints_used_as_labels_only": True,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "smoothness_validity_proxy_not_true_physical_validity": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = x._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["deploy_interaction_risk_proxy_head"] is True
    assert gate["deploy_true_physical_validity"] is False


def test_gate_rejects_weak_interaction_signal() -> None:
    payload = {
        "source": x.SOURCE,
        "stage43_w_precondition": {"verdict": "stage43_w_density_proxy_repaired_validity_proxy_diagnostic"},
        "training_protocol": {
            "checkpoint_sha256_matches_stage43_m": True,
            "selection_data": "train_val_selected_test_once",
            "test_threshold_tuning": False,
        },
        "interaction_risk_head": {
            "selected": {
                "test_metrics": {
                    "defined": True,
                    "auroc": 0.55,
                    "auprc": 0.5,
                    "positive_rate": 0.1,
                    "ece": 0.05,
                }
            }
        },
        "smoothness_validity_proxy_head": {
            "selected": {"test_metrics": {"rows": 100, "r2": 0.1, "corr": 0.4}}
        },
        "no_leakage": {
            "future_waypoint_input": False,
            "future_waypoints_used_as_labels_only": True,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "smoothness_validity_proxy_not_true_physical_validity": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = x._gate(payload)
    assert gate["passed"] < gate["total"]
    assert gate["deploy_interaction_risk_proxy_head"] is False
