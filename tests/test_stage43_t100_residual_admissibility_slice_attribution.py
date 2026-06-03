from __future__ import annotations

import numpy as np

from src import stage43_t100_residual_admissibility_slice_attribution as cv


def test_slice_table_tracks_gain_share() -> None:
    labels = np.asarray(["a", "a", "b", "b"])
    floor = np.asarray([10.0, 10.0, 10.0, 10.0])
    selected = np.asarray([9.0, 10.0, 8.0, 10.0])
    switched = np.asarray([True, False, True, False])
    rows = cv._slice_table(labels, selected, floor, switched)
    by_label = {row["label"]: row for row in rows}
    assert by_label["b"]["positive_gain_share"] > by_label["a"]["positive_gain_share"]
    assert by_label["a"]["switched"] == 1
    assert by_label["b"]["slice_improvement_vs_floor"] > by_label["a"]["slice_improvement_vs_floor"]


def test_aggregate_marks_narrow_when_one_source_dominates() -> None:
    runs = [
        {
            "max_metric_replay_diff": 0.0,
            "switch_rate": 0.05,
            "max_source_positive_gain_share": 0.95,
            "max_scene_positive_gain_share": 0.40,
            "positive_sources": 1,
            "switched_sources": 2,
            "concentration": {"source_signal_narrow": True, "scene_signal_narrow": False},
        }
        for _ in range(3)
    ]
    agg = cv._aggregate(runs)
    assert agg["all_replay_exact"] is True
    assert agg["scope_verdict"] == "narrow_supported_diagnostic"
    assert agg["any_seed_source_narrow"] is True


def test_gate_requires_exact_replay() -> None:
    payload = {
        "stage43_cu_precondition": {"verdict": "stage43_cu_t100_admissibility_multiseed_confirmed_tiny_positive"},
        "result_source": "fresh_t100_residual_admissibility_slice_attribution",
        "seed_attribution": [
            {
                "slice_tables": {
                    "domain": [{"label": "UCY"}],
                    "source_file": [{"label": "src"}],
                    "scene_id": [{"label": "scene"}],
                }
            }
            for _ in range(3)
        ],
        "aggregate": {"all_replay_exact": False, "scope_verdict": "narrow_supported_diagnostic"},
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
    gate = cv._gate(payload)
    assert gate["passed"] == gate["total"] - 1
    assert gate["verdict"] == "stage43_cv_t100_slice_attribution_incomplete_keep_floor"
