import numpy as np

from src import stage42_source_level_graph_context as s42as


def test_last_valid_motion_uses_last_valid_history_only():
    seq = np.zeros((2, 4, 7), dtype=np.float32)
    seq[0, 0, 2] = 1.0
    seq[0, 0, 4] = 0.0
    seq[0, 0, 6] = 1.0
    seq[0, 2, 2] = 2.0
    seq[0, 2, 4] = np.pi / 2
    seq[0, 2, 6] = 1.0
    out = s42as._last_valid_motion(seq)
    assert np.isclose(out["speed"][0], 2.0)
    assert abs(out["vx"][0]) < 1e-5
    assert np.isclose(out["vy"][0], 2.0, atol=1e-5)
    assert out["valid"][0]
    assert not out["valid"][1]


def test_build_graph_features_excludes_self_and_counts_neighbors():
    data = {
        "horizon": np.asarray([50, 50, 50], dtype=np.int16),
        "history_seq": np.zeros((3, 2, 7), dtype=np.float32),
        "source_file": np.asarray(["a", "a", "a"]),
        "frame_id": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        "current_x": np.asarray([0.0, 1.0, 4.0], dtype=np.float32),
        "current_y": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.ones(3, dtype=np.float32),
        "agent_id": np.asarray([1, 2, 3], dtype=np.int64),
    }
    data["history_seq"][:, -1, 2] = 1.0
    data["history_seq"][:, -1, 6] = 1.0
    features, names, stats = s42as._build_graph_features(data)
    assert features.shape[0] == 3
    assert features.shape[1] == len(names)
    assert stats["rows_with_neighbors"] == 3
    assert np.all(features[:, 0] == 2.0)
    assert np.isclose(features[0, names.index("graph_min_dist_norm")], 1.0)


def test_build_context_adds_goal_history_controls():
    n = 4
    data = {
        "dataset": np.asarray(["A", "B", "A", "B"]),
        "horizon": np.asarray([10, 25, 50, 100]),
        "history_seq": np.zeros((n, 2, 7), dtype=np.float32),
        "source_file": np.asarray(["a", "a", "b", "b"]),
        "frame_id": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "current_x": np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "current_y": np.zeros(n, dtype=np.float32),
        "scale": np.ones(n, dtype=np.float32),
        "agent_id": np.asarray([1, 2, 1, 2], dtype=np.int64),
        "prototype_likelihood": np.ones((n, 8), dtype=np.float32),
        "prototype_entropy": np.ones(n, dtype=np.float32),
        "goal_ambiguity": np.ones(n, dtype=np.float32),
        "history_scalar": np.ones((n, 9), dtype=np.float32),
    }
    graph, graph_names, _ = s42as._build_context(data, "graph_only")
    full, full_names, _ = s42as._build_context(data, "graph_history_goal")
    assert full.shape[0] == graph.shape[0]
    assert full.shape[1] > graph.shape[1]
    assert "prototype_entropy" in full_names
    assert "history_scalar_0" in full_names
    assert len(graph_names) == graph.shape[1]


def test_gate_requires_graph_increment():
    metric = {"all_improvement": 0.2, "t50_improvement": 0.1, "hard_failure_improvement": 0.15, "easy_degradation": 0.0}
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "graph_schema": {"feature_names": s42as._knn_feature_names(), "uses_current_and_past_only": True},
        "context_stats": {"a": {"feature_count": len(s42as._knn_feature_names()), "rows_with_neighbors": 10, "uses_future_endpoint": False, "uses_future_waypoint": False}},
        "baseline_family_only": {
            "protected": metric,
            "bootstrap": {"all": {"bootstrap_n": 1000}, "t50": {"bootstrap_n": 1000}},
        },
        "graph_variants": {"a": {}, "b": {}, "c": {}},
        "positive_graph_context_variants": [],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "graph_features_current_and_past_only": True,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42as._gate(result)
    assert gate["passed"] == gate["total"] - 1
    assert gate["gates"]["graph_context_increment_found"] is False
