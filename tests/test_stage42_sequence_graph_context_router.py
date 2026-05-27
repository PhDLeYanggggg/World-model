from __future__ import annotations

import numpy as np

from src.stage42_sequence_graph_context_router import _augmented_router_features, _gate, _sequence_summary
from src.stage42_context_gain_router import _train_gain_router


def test_stage42_eq_sequence_summary_uses_past_history_shape() -> None:
    n = 5
    k = 4
    seq = np.zeros((n, k, 7), dtype=np.float32)
    seq[..., 6] = 1.0
    seq[:, :, 0] = np.arange(k, dtype=np.float32)[None, :]
    seq[:, :, 1] = np.arange(k, dtype=np.float32)[None, :] * 0.5
    seq[:, :, 2] = np.linspace(0.1, 0.4, k, dtype=np.float32)[None, :]
    seq[:, :, 4] = 0.25
    features, names, stats = _sequence_summary({"history_seq": seq})

    assert features.shape == (n, len(names))
    assert stats["uses_future_endpoint"] is False
    assert stats["uses_future_waypoint"] is False
    assert stats["valid_history_min"] == k
    assert np.all(features[:, names.index("seq_path_length")] > 0)


def test_stage42_eq_augmented_router_features_concatenates_context() -> None:
    candidate = np.ones((3, 2), dtype=np.float32)
    graph = np.ones((3, 4), dtype=np.float32) * 2
    seq = np.ones((3, 5), dtype=np.float32) * 3
    augmented = _augmented_router_features(candidate, graph, seq)

    assert augmented.shape == (3, 11)
    assert np.allclose(augmented[:, :2], 1.0)
    assert np.allclose(augmented[:, 2:6], 2.0)
    assert np.allclose(augmented[:, 6:], 3.0)


def test_stage42_eq_gain_router_can_use_augmented_signal_safely() -> None:
    rng = np.random.default_rng(42)
    n = 300
    weak_candidate = rng.normal(size=(n, 2)).astype(np.float32)
    graph_signal = rng.normal(size=(n, 1)).astype(np.float32)
    seq_signal = rng.normal(size=(n, 1)).astype(np.float32)
    x = _augmented_router_features(weak_candidate, graph_signal, seq_signal)
    split = np.array(["train"] * 150 + ["val"] * 75 + ["test"] * 75)
    helpful = (graph_signal[:, 0] + seq_signal[:, 0]) > 0.35
    base = np.ones(n, dtype=np.float64)
    candidate = base.copy()
    candidate[helpful] -= 0.30
    candidate[~helpful] += 0.04
    data = {
        "horizon": np.array([50] * n),
        "hard": helpful,
        "failure": helpful,
        "easy": ~helpful,
    }

    result = _train_gain_router(
        name="synthetic_sequence_graph_context",
        raw_router_features=x,
        base_ade=base,
        candidate_ade=candidate,
        split=split,
        data=data,
    )

    assert result["validation_selection"]["source"] == "validation_only"
    assert result["test_metric_vs_baseline_family"]["all_improvement"] > 0
    assert result["test_metric_vs_baseline_family"]["easy_degradation"] <= 0.02


def test_stage42_eq_gate_passes_for_bounded_negative_or_positive_result() -> None:
    payload = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "baseline_family_control": {"policy_slice_count": 1},
        "sequence_summary_schema": {"stats": {"feature_count": 11}},
        "graph_summary_schema": {"stats": {"rows_with_neighbors": 10}},
        "routers": {
            f"r{i}": {"validation_selection": {"source": "validation_only", "test_threshold_tuning": False}}
            for i in range(3)
        },
        "summary": {"sequence_graph_increment_verdict": "stage42_eq_sequence_graph_context_router_not_supported"},
        "positive_sequence_graph_context_routers": [],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "sequence_summary_current_past_only": True,
            "graph_summary_current_past_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_selected_thresholds": True,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
