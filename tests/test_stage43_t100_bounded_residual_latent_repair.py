from __future__ import annotations

import numpy as np
import torch

from src import stage43_t100_bounded_residual_latent_repair as cs


def test_residual_model_output_is_component_bounded() -> None:
    model = cs.BoundedResidualLatentDynamics(input_dim=5, hidden_dim=8, latent_dim=4, residual_clip=0.12)
    out = model(torch.zeros((3, 5), dtype=torch.float32))
    residual = out["residual_delta"].detach().numpy()
    assert residual.shape == (3, 4, 2)
    assert float(np.max(np.abs(residual))) <= 0.120001


def test_evaluate_policy_can_force_floor_on_easy_rows() -> None:
    ds = type("Dummy", (), {})()
    ds.x = np.zeros((4, 2), dtype=np.float32)
    ds.floor_ade = np.ones(4, dtype=np.float32)
    ds.floor_fde = np.ones(4, dtype=np.float32)
    ds.floor_waypoint_delta = np.zeros((4, 4, 2), dtype=np.float32)
    ds.waypoint_delta = np.zeros((4, 4, 2), dtype=np.float32)
    ds.waypoint_valid = np.ones((4, 4), dtype=bool)
    ds.hard = np.zeros(4, dtype=bool)
    ds.failure = np.zeros(4, dtype=bool)
    ds.easy = np.ones(4, dtype=bool)
    ds.horizon = np.asarray([100, 100, 100, 100])
    pred = {
        "residual": np.ones((4, 4, 2), dtype=np.float32),
        "gain": np.ones(4, dtype=np.float32),
        "harm": np.zeros(4, dtype=np.float32),
        "failure": np.ones(4, dtype=np.float32),
    }
    result = cs._evaluate_policy(
        ds,
        pred,
        {"alpha": 1.0, "policy": {"gain_threshold": 0.0, "harm_threshold": 1.0, "failure_threshold": 0.0}, "force_easy_floor": True},
    )
    assert result["metrics"]["switch_rate"] == 0.0
    assert result["metrics"]["easy_degradation_vs_floor"] == 0.0


def test_gate_accepts_honest_floor_with_bounded_residual() -> None:
    payload = {
        "stage43_cq_precondition": {"verdict": "stage43_cq_t100_source_scene_supported_supervision_cache_pass"},
        "result_source": "fresh_torch_t100_bounded_residual_latent_repair",
        "checkpoint": "README_RESULTS.md",
        "checkpoint_committed": False,
        "horizon_protocol": {"horizons": [100]},
        "feature_contract": {"denied_feature_name_hits": []},
        "residual_bounds": {"max_abs_predicted_residual": 0.1, "residual_clip": 0.2},
        "latent_variance": 0.05,
        "selection_protocol": {"test_threshold_tuning": False},
        "test_metrics_with_floor": {
            "rows": 10,
            "easy_degradation_vs_floor": 0.0,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.0,
            "switch_rate": 0.0,
        },
        "test_metrics_neural_without_floor": {"easy_degradation_vs_floor": 0.2},
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
    gate = cs._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_cs_t100_bounded_residual_latent_keep_floor"
