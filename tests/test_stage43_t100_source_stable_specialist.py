from __future__ import annotations

import numpy as np

from src import stage43_t100_source_stable_specialist as t


def test_source_subset_filters_h100_and_source_names() -> None:
    class Fake:
        split = "pool"
        x = np.arange(12, dtype=np.float32).reshape(4, 3)
        waypoint_delta = np.zeros((4, 4, 2), dtype=np.float32)
        waypoint_valid = np.ones((4, 4), dtype=bool)
        floor_waypoint_delta = np.zeros((4, 4, 2), dtype=np.float32)
        floor_ade = np.ones(4, dtype=np.float32)
        floor_fde = np.ones(4, dtype=np.float32)
        y_failure = np.zeros(4, dtype=np.float32)
        y_gain = np.zeros(4, dtype=np.float32)
        y_harm = np.zeros(4, dtype=np.float32)
        y_density = np.zeros(4, dtype=np.float32)
        horizon = np.asarray([100, 50, 100, 100])
        domain = np.asarray(["d"] * 4)
        source_file = np.asarray([
            "/root/external_data/OpenTraj/a.txt",
            "/root/external_data/OpenTraj/a.txt",
            "/root/external_data/OpenTraj/b.txt",
            "/root/external_data/OpenTraj/c.txt",
        ])
        scene_id = np.asarray(["s"] * 4)
        hard = np.zeros(4, dtype=bool)
        failure = np.zeros(4, dtype=bool)
        easy = np.zeros(4, dtype=bool)
        scale = np.ones(4, dtype=np.float32)
        feature_names = ["a", "b", "c"]

    ds = t._subset(Fake(), ["OpenTraj/a.txt", "OpenTraj/c.txt"], "test", horizon=100)
    assert ds.split == "test"
    assert len(ds.x) == 2
    assert ds.x[:, 0].tolist() == [0.0, 9.0]


def test_gate_marks_positive_but_unsafe_as_not_deployed() -> None:
    payload = {
        "source": t.SOURCE,
        "stage43_s_precondition": {
            "verdict": "stage43_s_t100_source_coverage_preflight_pass",
            "feasible_families": [t.FAMILY],
        },
        "source_level_split": {"train_rows": 10, "val_rows": 5, "test_rows": 5},
        "training_protocol": {
            "selection_data": "source_level_validation_only",
            "test_threshold_tuning": False,
            "max_easy_degradation": 0.02,
        },
        "no_leakage": {
            "test_threshold_tuning": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
        },
        "source_stable_h100_test_metrics": {
            "rows": 5,
            "full_waypoint_ade_improvement_vs_floor": 0.05,
            "easy_degradation_vs_floor": 0.15,
        },
        "selected_specialist": {
            "validation_source_safety": {
                "source_safe": True,
            },
        },
        "deployment": {"deploy_source_stable_h100_specialist": False, "reason": "candidate_positive_but_easy_harm_exceeds_guard"},
        "deployment_metrics": {"easy_degradation_vs_floor": 0.0},
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = t._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["positive_h100_dynamics_signal"] is True
    assert gate["easy_safe"] is False
    assert gate["deploy_source_stable_h100_specialist"] is False
