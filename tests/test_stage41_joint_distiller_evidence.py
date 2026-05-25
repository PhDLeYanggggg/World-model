import numpy as np

from src import stage41_joint_distiller_evidence as jde


def test_feature_slices_cover_feature_matrix_once():
    groups = jde._feature_slices(32)
    atomic = [
        "static_causal_features",
        "full_trajectory_prediction_signals",
        "current_group_geometry",
        "floor_group_geometry",
        "neural_group_geometry",
        "neighbor_count",
        "domain_embedding",
        "horizon_embedding",
    ]
    covered = sorted(i for name in atomic for i in groups[name])
    assert covered == list(range(32))
    assert groups["all_group_geometry"] == groups["current_group_geometry"] + groups["floor_group_geometry"] + groups["neural_group_geometry"]


def test_bootstrap_ci_positive_for_consistent_gain():
    fallback = np.ones(200, dtype=float)
    selected = np.full(200, 0.8, dtype=float)
    ci = jde._bootstrap_ci(selected, fallback, np.ones(200, dtype=bool), n=100, seed=7)
    assert ci["n"] == 200
    assert ci["low"] > 0.15
    assert ci["high"] < 0.25
