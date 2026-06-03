from __future__ import annotations

import numpy as np

from src import stage43_t100_residual_admissibility_group_stress as cx


def test_label_gain_rows_orders_positive_gain() -> None:
    labels = np.asarray(["a", "a", "b", "b"])
    floor = np.asarray([10.0, 10.0, 10.0, 10.0])
    selected = np.asarray([9.0, 10.0, 7.0, 10.0])
    switched = np.asarray([True, False, True, False])
    rows = cx._label_gain_rows(labels, floor, selected, switched)
    assert rows[0]["label"] == "b"
    assert rows[0]["positive_gain_sum"] > rows[1]["positive_gain_sum"]


def test_aggregate_marks_group_survival() -> None:
    runs = [
        {
            "max_metric_replay_diff": 0.0,
            "summary": {
                "min_without_any_group_t100": 0.001,
                "source_group_flip_count": 0,
                "scene_group_flip_count": 0,
                "domain_pair_flip_count": 0,
                "group_count": 6,
            },
        }
        for _ in range(3)
    ]
    agg = cx._aggregate(runs)
    assert agg["all_replay_exact"] is True
    assert agg["all_group_exclusions_positive"] is True
    assert agg["group_stress_verdict"] == "multi_source_group_stress_survives"


def test_gate_requires_group_tables() -> None:
    payload = {
        "stage43_cw_precondition": {"verdict": "stage43_cw_t100_source_stress_survives_single_exclusion_diagnostic"},
        "result_source": "fresh_t100_multi_source_group_stress",
        "seed_group_stress": [
            {"group_stress_tables": {"source_groups": [], "scene_groups": [], "domain_pair_groups": []}}
            for _ in range(3)
        ],
        "aggregate": {"all_replay_exact": True, "group_stress_verdict": "multi_source_group_stress_survives"},
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
    gate = cx._gate(payload)
    assert gate["passed"] == gate["total"] - 1
    assert gate["verdict"] == "stage43_cx_t100_group_stress_incomplete_keep_floor"
