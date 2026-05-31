from __future__ import annotations

import numpy as np

from src import stage43_full_waypoint_latent_dynamics as m


def _payload(metrics: dict, *, deploy: bool = True) -> dict:
    return {
        "source": m.SOURCE,
        "result_source": "fresh_run",
        "checkpoint": "dummy.pt",
        "checkpoint_committed": False,
        "stage43_l_precondition": {"full_waypoint_supervised_training_ready": True},
        "no_leakage": {
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity_input": False,
        },
        "latent_variance": 0.05,
        "test_metrics_with_floor": metrics,
        "test_metrics_neural_without_floor": dict(metrics),
        "deploy_neural": deploy,
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_trajectory_error_zero_for_identical_waypoints() -> None:
    ds = m.WaypointSplit(
        split="test",
        x=np.zeros((2, 3), dtype=np.float32),
        waypoint_delta=np.zeros((2, 4, 2), dtype=np.float32),
        waypoint_valid=np.ones((2, 4), dtype=bool),
        floor_waypoint_delta=np.zeros((2, 4, 2), dtype=np.float32),
        floor_ade=np.ones(2, dtype=np.float32),
        floor_fde=np.ones(2, dtype=np.float32),
        y_failure=np.zeros(2, dtype=np.float32),
        y_gain=np.zeros(2, dtype=np.float32),
        y_harm=np.zeros(2, dtype=np.float32),
        y_density=np.zeros(2, dtype=np.float32),
        horizon=np.asarray([50, 100]),
        domain=np.asarray(["UCY", "TrajNet"]),
        source_file=np.asarray(["a", "b"]),
        scene_id=np.asarray(["s", "t"]),
        hard=np.asarray([True, False]),
        failure=np.asarray([False, False]),
        easy=np.asarray([False, True]),
        scale=np.ones(2, dtype=np.float32),
        feature_names=["a", "b", "c"],
    )
    ade, fde = m._trajectory_error(ds, ds.waypoint_delta.copy())
    assert np.allclose(ade, 0.0)
    assert np.allclose(fde, 0.0)


def test_gate_accepts_protected_positive_full_waypoint_candidate(monkeypatch) -> None:
    monkeypatch.setattr(m.Path, "exists", lambda self: True)
    metrics = {
        "rows": 10,
        "full_waypoint_ade_improvement_vs_floor": 0.03,
        "endpoint_fde_improvement_vs_floor": 0.01,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.04,
        "t50_endpoint_fde_improvement_vs_floor": 0.02,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.05,
        "easy_degradation_vs_floor": 0.01,
        "switch_rate": 0.2,
        "harm_over_floor_ade": -0.1,
        "mean_floor_ade": 1.0,
        "mean_selected_ade": 0.9,
    }
    gate = m._gate(_payload(metrics, deploy=True))
    assert gate["passed"] == gate["total"]
    assert gate["deploy_neural_full_waypoint"] is True


def test_gate_blocks_easy_harm_deployment(monkeypatch) -> None:
    monkeypatch.setattr(m.Path, "exists", lambda self: True)
    metrics = {
        "rows": 10,
        "full_waypoint_ade_improvement_vs_floor": 0.05,
        "endpoint_fde_improvement_vs_floor": 0.04,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.05,
        "t50_endpoint_fde_improvement_vs_floor": 0.03,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.08,
        "easy_degradation_vs_floor": 0.03,
        "switch_rate": 0.2,
        "harm_over_floor_ade": -0.1,
        "mean_floor_ade": 1.0,
        "mean_selected_ade": 0.9,
    }
    gate = m._gate(_payload(metrics, deploy=True))
    assert gate["gates"]["easy_preserved"] is False
    assert gate["deploy_neural_full_waypoint"] is False
