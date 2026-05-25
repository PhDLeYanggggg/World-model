import numpy as np

from src import stage41_group_consistency_evidence as gce


def test_feature_slices_add_group_blocks():
    groups = gce._feature_slices(42)
    assert len(groups["group_consistency_features"]) == 10
    assert len(groups["group_consistency_geometry"]) == 6
    assert len(groups["proposal_score_features"]) == 4
    assert max(groups["group_consistency_features"]) == 41


def test_selected_from_policy_uses_safe_gain_unsafe_and_proposal():
    scores = {
        "safe_prob": np.asarray([0.8, 0.8, 0.3, 0.9]),
        "gain_pred": np.asarray([0.5, -0.1, 0.5, 0.5]),
        "unsafe_prob": np.asarray([0.1, 0.1, 0.1, 0.9]),
    }
    data = {
        "proposal_switch": np.asarray([True, True, True, False]),
        "floor_ade": np.ones(4),
        "neural_ade": np.asarray([0.4, 0.2, 0.2, 0.2]),
    }
    selected, switch = gce._selected_from_policy(scores, data, {"safe_min": 0.5, "gain_min": 0.0, "unsafe_max": 0.5})
    assert switch.tolist() == [True, False, False, False]
    assert np.allclose(selected, [0.4, 1.0, 1.0, 1.0])


def test_bootstrap_ci_positive_when_selected_better():
    selected = np.full(100, 0.5)
    fallback = np.ones(100)
    ci = gce._bootstrap_ci(selected, fallback, np.ones(100, dtype=bool), n=100, seed=7)
    assert ci["low"] > 0.4
    assert ci["bootstrap_n"] == 100
