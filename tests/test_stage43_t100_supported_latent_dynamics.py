from __future__ import annotations

import numpy as np

from src import stage43_t100_supported_latent_dynamics as cr


def test_feature_contract_rejects_future_names() -> None:
    clean = cr._feature_contract(["history_speed", "baseline_endpoint_rel_0"])
    assert clean["denied_feature_name_hits"] == []
    dirty = cr._feature_contract(["future_endpoint_x", "history_speed"])
    assert dirty["denied_feature_name_hits"] == ["future_endpoint_x"]


def test_gate_keeps_current_heldout_unchanged() -> None:
    payload = {
        "stage43_cq_precondition": {"verdict": "stage43_cq_t100_source_scene_supported_supervision_cache_pass"},
        "result_source": "fresh_torch_t100_supported_latent_dynamics",
        "checkpoint": "README_RESULTS.md",
        "checkpoint_committed": False,
        "horizon_protocol": {"horizons": [100]},
        "feature_contract": {"denied_feature_name_hits": []},
        "latent_variance": 0.05,
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
    gate = cr._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_cr_t100_supported_latent_dynamics_keep_floor"


def test_search_t100_policy_can_select_floor_on_unsafe_predictions() -> None:
    ds = type("Dummy", (), {})()
    ds.x = np.zeros((4, 2), dtype=np.float32)
    ds.floor_ade = np.ones(4, dtype=np.float32)
    ds.floor_fde = np.ones(4, dtype=np.float32)
    ds.hard = np.zeros(4, dtype=bool)
    ds.failure = np.zeros(4, dtype=bool)
    ds.easy = np.ones(4, dtype=bool)
    ds.horizon = np.asarray([100, 100, 100, 100])
    ds.waypoint_delta = np.zeros((4, 4, 2), dtype=np.float32)
    ds.waypoint_valid = np.ones((4, 4), dtype=bool)
    pred = {
        "waypoint": np.ones((4, 4, 2), dtype=np.float32) * 10.0,
        "gain": np.zeros(4, dtype=np.float32),
        "harm": np.ones(4, dtype=np.float32),
        "failure": np.zeros(4, dtype=np.float32),
    }
    policy = cr._search_t100_policy(ds, pred)
    assert policy["validation_metrics"]["easy_degradation_vs_floor"] == 0.0
