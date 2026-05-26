from __future__ import annotations

import numpy as np

from src import stage42_full_waypoint_proximity_occupancy_loss_repair as dh


def test_graph_signals_detect_close_and_closing_neighbors() -> None:
    graph = np.zeros((2, 34), dtype=np.float32)
    graph[:, 1] = [0.4, 2.0]
    graph[:, 3] = [0.71, 0.33]
    graph[:, 4] = [1.0, 0.0]
    graph[:, 5] = [1.0, 0.5]
    graph[:, 6] = [1.0, 0.5]
    graph[:, 13] = [2.0, -1.0]
    sig = dh._graph_signals(graph)
    assert sig["close"].tolist() == [1.0, 0.0]
    assert sig["very_close"].tolist() == [1.0, 0.0]
    assert sig["mean_closing"][0] > sig["mean_closing"][1]


def test_loss_weights_emphasize_proximity_and_hard_long() -> None:
    data = {
        "horizon": np.asarray([10, 50, 100]),
        "hard": np.asarray([False, True, False]),
        "failure": np.asarray([False, False, True]),
        "easy": np.asarray([True, False, False]),
    }
    graph = np.zeros((3, 34), dtype=np.float32)
    graph[:, 1] = [0.3, 2.0, 0.8]
    graph[:, 3] = [0.75, 0.25, 0.55]
    graph[:, 4] = [1.0, 0.0, 1.0]
    graph[:, 5] = [1.0, 0.0, 1.0]
    train = np.asarray([True, True, True])
    prox = dh._loss_weights(data, graph, train, "proximity_close_weighted")
    hard_long = dh._loss_weights(data, graph, train, "proximity_hard_long_weighted")
    assert prox[0] > prox[1]
    assert hard_long[2] > hard_long[0]


def test_gate_reports_positive_not_better_than_am() -> None:
    result = {
        "split_stats": {"by_split": {"test": {"rows": 5}}},
        "label_stats": {"test_full_waypoint_rows": 5},
        "graph_stats": {"rows_with_neighbors": 5},
        "feature_schema": {"graph_feature_count": 4, "variants": ["a", "b"]},
        "model": {
            "candidate_count": 2,
            "selected": {"val_score": 1.0},
            "metrics": {
                "protected_selected_candidate": {
                    "rows": 5,
                    "all_improvement": 0.1,
                    "hard_failure_improvement": 0.1,
                    "easy_degradation": 0.0,
                }
            },
            "bootstrap": {"all": {"bootstrap_n": 1000}},
        },
        "comparison_to_stage42_am": {
            "delta_vs_stage42_am": {"all_improvement": -0.01, "hard_failure_improvement": -0.01}
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "graph_features_current_and_past_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "validation_only_model_selection": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = dh._gate(result)
    assert gate["gates"]["graph_proximity_features_built"] is True
    assert gate["gates"]["beats_stage42_am_all"] is False
    assert gate["verdict"] == "stage42_dh_proximity_occupancy_loss_repair_pass_positive_not_better_than_am"
