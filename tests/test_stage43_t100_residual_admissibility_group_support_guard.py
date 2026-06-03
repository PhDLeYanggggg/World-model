from __future__ import annotations

import numpy as np

from src import stage43_t100_residual_admissibility_group_support_guard as cy


def test_guard_mask_variants() -> None:
    class Dummy:
        source_file = np.asarray(["s1", "s2", "s1"])
        scene_id = np.asarray(["a", "a", "b"])
        domain = np.asarray(["d", "e", "d"])
        x = np.zeros((3, 2), dtype=np.float32)

    eligible = {"source": ["s1"], "scene": ["a"], "domain": ["d"]}
    assert cy._guard_mask(Dummy, "source_val_positive", eligible).tolist() == [True, False, True]
    assert cy._guard_mask(Dummy, "scene_val_positive", eligible).tolist() == [True, True, False]
    assert cy._guard_mask(Dummy, "domain_and_source_and_scene", eligible).tolist() == [True, False, False]
    assert cy._guard_mask(Dummy, "floor", eligible).tolist() == [False, False, False]


def test_variant_objective_penalizes_fragility() -> None:
    metrics = {
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.01,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.02,
        "full_waypoint_ade_improvement_vs_floor": 0.01,
        "easy_degradation_vs_floor": 0.0,
        "switch_rate": 0.1,
    }
    stable = {"min_without_any_group_t100": 0.002, "source_group_flip_count": 0, "scene_group_flip_count": 0, "domain_pair_flip_count": 0}
    fragile = {"min_without_any_group_t100": -0.002, "source_group_flip_count": 0, "scene_group_flip_count": 1, "domain_pair_flip_count": 0}
    assert cy._variant_objective(metrics, stable) > cy._variant_objective(metrics, fragile)


def test_aggregate_marks_reduction() -> None:
    runs = [
        {
            "max_replay_diff": 0.0,
            "selected_variant": {"variant": "source_val_positive"},
            "base_test_metrics": {"t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.001},
            "guarded_test_metrics": {
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0008,
                "easy_degradation_vs_floor": 0.0,
                "switch_rate": 0.05,
            },
            "base_test_group_summary": {"min_without_any_group_t100": -0.001},
            "guarded_test_group_summary": {"min_without_any_group_t100": 0.0002},
            "test_delta_vs_base": {"t100": -0.0002, "min_without_group_t100": 0.0012},
        }
        for _ in range(3)
    ]
    agg = cy._aggregate(runs)
    assert agg["all_replay_exact"] is True
    assert agg["group_fragility_reduced"] is True
    assert agg["guard_preserves_easy"] is True
