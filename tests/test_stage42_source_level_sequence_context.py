import numpy as np

from src import stage42_source_level_sequence_context as s42ar


def test_build_context_shapes_and_variant_zeroing():
    data = {
        "dataset": np.asarray(["A", "B"]),
        "horizon": np.asarray([10, 50]),
        "history_scalar": np.ones((2, 9), dtype=np.float32),
        "prototype_likelihood": np.ones((2, 8), dtype=np.float32),
        "prototype_entropy": np.ones(2, dtype=np.float32),
        "goal_ambiguity": np.ones(2, dtype=np.float32),
    }
    hist_ctx = s42ar._build_context(data, "sequence_history")
    full_ctx = s42ar._build_context(data, "sequence_history_goal_neighbor")
    assert hist_ctx.shape == full_ctx.shape
    assert np.all(hist_ctx[:, :15] == 0.0)
    assert np.any(full_ctx[:, :15] != 0.0)


def test_build_sequence_zeroes_history_for_goal_neighbor_variant():
    data = {"history_seq": np.ones((2, 4, 7), dtype=np.float32)}
    seq, valid = s42ar._build_sequence(data, "sequence_goal_neighbor_no_history")
    assert np.all(seq == 0.0)
    assert np.all(valid == 0.0)
    seq2, valid2 = s42ar._build_sequence(data, "sequence_history")
    assert np.all(seq2 == 1.0)
    assert np.all(valid2 == 1.0)


def test_standardize_sequence_uses_train_only_stats():
    seq = np.asarray([[[1.0, 2.0]], [[3.0, 4.0]], [[100.0, 200.0]]], dtype=np.float32)
    train = np.asarray([True, True, False])
    z, mean, std = s42ar._standardize_sequence(seq, train)
    assert np.allclose(mean, [2.0, 3.0])
    assert np.allclose(std, [1.0, 1.0])
    assert np.allclose(z[0, 0], [-1.0, -1.0])


def test_gate_requires_sequence_increment():
    metric = {"all_improvement": 0.2, "t50_improvement": 0.1, "hard_failure_improvement": 0.15, "easy_degradation": 0.0}
    result = {
        "runtime": {"machine": "arm64", "num_workers": 0},
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "baseline_family_only": {
            "protected": metric,
            "bootstrap": {"all": {"bootstrap_n": 1000}, "t50": {"bootstrap_n": 1000}},
        },
        "sequence_variants": {"a": {}, "b": {}, "c": {}},
        "positive_sequence_context_variants": [],
        "training": {
            "a": {"checkpoint": "README_RESULTS.md"},
            "b": {"checkpoint": "README_RESULTS.md"},
            "c": {"checkpoint": "README_RESULTS.md"},
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
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
    gate = s42ar._gate(result)
    assert gate["passed"] == gate["total"] - 1
    assert gate["gates"]["sequence_context_increment_found"] is False
