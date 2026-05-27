from __future__ import annotations

import numpy as np

from src import stage42_proximity_pareto_composer as fb


def test_group_any_mask_marks_whole_group_when_any_row_risky() -> None:
    keys = np.asarray(["a", "a", "b", "b"], dtype=object)
    risk = np.asarray([False, True, False, False])
    out = fb._group_any_mask(keys, risk)
    assert out.tolist() == [True, True, False, False]


def test_delta_reports_difference_against_reference() -> None:
    metric = {"all_improvement": 0.2, "t50_improvement": 0.1, "t100_raw_frame_diagnostic_improvement": 0.0, "hard_failure_improvement": 0.3, "easy_degradation": 0.01}
    ref = {"all_improvement": 0.15, "t50_improvement": 0.12, "t100_raw_frame_diagnostic_improvement": 0.0, "hard_failure_improvement": 0.25, "easy_degradation": 0.02}
    delta = fb._delta(metric, ref)
    assert np.isclose(delta["all_improvement"], 0.05)
    assert np.isclose(delta["t50_improvement"], -0.02)
    assert np.isclose(delta["hard_failure_improvement"], 0.05)
    assert np.isclose(delta["easy_degradation"], -0.01)


def test_candidate_grid_contains_di_reference_and_group_modes() -> None:
    grid = fb._candidate_grid()
    modes = {row["mode"] for row in grid}
    assert "all_di" in modes
    assert "row_di_near" in modes
    assert "group_di_near" in modes
    assert "group_di_near_fa_safer" in modes
    assert len(grid) >= 20


def test_gate_promotes_when_near_improves_without_material_di_loss() -> None:
    result = {
        "split_stats": {"by_split": {"test": {"rows": 10}}},
        "label_stats": {"test_full_waypoint_rows": 10},
        "composer_family": {"candidate_count": 20},
        "repair": {
            "selected": {"val_score": 1.0},
            "test": {
                "metric_vs_floor": {
                    "rows": 10,
                    "all_improvement": 0.1,
                    "t50_improvement": 0.1,
                    "hard_failure_improvement": 0.1,
                    "easy_degradation": 0.0,
                },
                "delta_vs_di": {
                    "all_improvement": -0.0001,
                    "hard_failure_improvement": -0.0001,
                },
                "near_delta_vs_di": -0.001,
                "bootstrap": {"all": {"bootstrap_n": 1000}},
            },
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "group_features_predicted_rollout_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = fb._gate(result)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_fb_proximity_pareto_composer_pass_promotable"
