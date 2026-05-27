import numpy as np

from src import stage42_constrained_fc_safety_composer as fe


def test_candidate_grid_contains_fc_and_safety_fallbacks() -> None:
    grid = fe._candidate_grid()
    modes = {row["mode"] for row in grid}
    fallbacks = {row["fallback"] for row in grid}

    assert "all_fc" in modes
    assert "fc_to_safety" in modes
    assert {"di", "fa", "fb", "safest"}.issubset(fallbacks)
    assert len(grid) >= 40


def test_select_validation_row_prefers_feasible_near_constraint() -> None:
    rows = [
        {
            "val_score": 10.0,
            "val_metric": {"all_improvement": 0.2, "hard_failure_improvement": 0.2, "easy_degradation": 0.0},
            "val_near_delta_vs_di": 0.1,
        },
        {
            "val_score": 1.0,
            "val_metric": {"all_improvement": 0.1, "hard_failure_improvement": 0.1, "easy_degradation": 0.0},
            "val_near_delta_vs_di": -0.01,
        },
    ]

    selected = fe._select_validation_row(rows)

    assert selected["val_score"] == 1.0


def test_group_any_delegates_whole_group_risk() -> None:
    keys = np.asarray(["g1", "g1", "g2"])
    risk = np.asarray([False, True, False])

    out = fe._group_any(keys, risk)

    assert out.tolist() == [True, True, False]


def test_gate_blocks_stage5c_and_smc() -> None:
    payload = {
        "source": fe.SOURCE,
        "split_stats": {"by_split": {"test": {"rows": 10}}},
        "label_stats": {"test_full_waypoint_rows": 10},
        "composer_family": {"fc_candidate_rebuilt": True},
        "repair": {
            "candidate_count": 40,
            "selected": {"val_score": 1.0},
            "test": {
                "metric_vs_floor": {
                    "rows": 10,
                    "all_improvement": 0.1,
                    "t50_improvement": 0.1,
                    "hard_failure_improvement": 0.1,
                    "easy_degradation": 0.0,
                },
                "delta_vs_fc": {"all_improvement": 0.0, "hard_failure_improvement": 0.0},
                "near_delta_vs_fc": -0.01,
                "near_delta_vs_di": 0.0,
                "bootstrap": {"all": {"bootstrap_n": 100}},
            },
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "composer_features_predicted_rollout_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fe._gate(payload)

    assert gate["gates"]["no_future_or_test_leakage"] is True
    assert gate["gates"]["stage5c_false"] is True
    assert gate["gates"]["smc_false"] is True
