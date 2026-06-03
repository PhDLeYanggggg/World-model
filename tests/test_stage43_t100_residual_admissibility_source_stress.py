from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src import stage43_t100_residual_admissibility_source_stress as cw


def _toy_ds() -> SimpleNamespace:
    return SimpleNamespace(
        floor_ade=np.asarray([10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        floor_fde=np.asarray([12.0, 12.0, 12.0, 12.0], dtype=np.float32),
        hard=np.asarray([True, False, True, False]),
        failure=np.asarray([False, False, True, False]),
        easy=np.asarray([False, True, False, True]),
    )


def test_masked_metrics_reports_easy_degradation() -> None:
    ds = _toy_ds()
    selected_ade = np.asarray([9.0, 11.0, 8.0, 10.0], dtype=np.float32)
    selected_fde = np.asarray([11.0, 13.0, 10.0, 12.0], dtype=np.float32)
    switched = np.asarray([True, True, True, False])
    metrics = cw._masked_metrics(ds, selected_ade, selected_fde, switched, np.asarray([True, True, True, True]))
    assert metrics["t100_improvement"] > 0.0
    assert metrics["hard_failure_improvement"] > 0.0
    assert metrics["easy_degradation"] > 0.0


def test_aggregate_detects_fragile_source_removal() -> None:
    runs = [
        {
            "max_metric_replay_diff": 0.0,
            "summary": {
                "min_without_source_t100": -0.01,
                "min_without_scene_t100": 0.002,
                "min_without_domain_t100": 0.001,
                "source_removal_flip_count": 1,
                "scene_removal_flip_count": 0,
                "domain_removal_flip_count": 0,
                "negative_source_slice_count": 2,
                "negative_scene_slice_count": 2,
                "negative_domain_slice_count": 0,
            },
        }
        for _ in range(3)
    ]
    agg = cw._aggregate(runs)
    assert agg["all_replay_exact"] is True
    assert agg["all_single_source_exclusions_positive"] is False
    assert agg["stress_verdict"] == "source_scene_stress_fragile_keep_diagnostic"


def test_gate_requires_replay_exact() -> None:
    payload = {
        "stage43_cv_precondition": {"verdict": "stage43_cv_t100_slice_attribution_broad_supported_diagnostic"},
        "result_source": "fresh_t100_source_scene_single_exclusion_stress",
        "seed_stress": [
            {
                "stress_tables": {
                    "source_file": [{"label": "src"}],
                    "scene_id": [{"label": "scene"}],
                    "domain": [{"label": "domain"}],
                }
            }
            for _ in range(3)
        ],
        "aggregate": {"all_replay_exact": False, "stress_verdict": "source_scene_stress_survives_single_exclusion"},
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
    gate = cw._gate(payload)
    assert gate["passed"] == gate["total"] - 1
    assert gate["verdict"] == "stage43_cw_t100_source_stress_incomplete_keep_floor"
