import numpy as np

from src import stage41_joint_latent_rollout as jlr


def test_group_indices_groups_matching_keys():
    keys = np.asarray(["b", "a", "b", "c", "a"], dtype=object)
    groups = jlr._group_indices(keys)
    assert [g.tolist() for g in groups] == [[1, 4], [0, 2], [3]]


def test_joint_future_close_label_marks_close_future_group():
    waypoint_xy = np.asarray(
        [
            [[[0.0, 0.0]], [[0.01, 0.0]], [[4.0, 0.0]]],
        ],
        dtype=float,
    ).reshape(3, 1, 2)
    keys = np.asarray(["g", "g", "g"], dtype=object)
    normalizer = np.ones(3, dtype=float)
    label = jlr._joint_future_close_label(waypoint_xy, keys, normalizer, threshold=0.05)
    assert label.tolist() == [1.0, 1.0, 0.0]


def test_policy_switch_uses_predicted_scores_only():
    pred = {
        "traj_risk": np.asarray([0.1, 0.8, 0.1]),
        "physical": np.asarray([0.9, 0.9, 0.1]),
        "future_close": np.asarray([0.1, 0.1, 0.1]),
    }
    switch = jlr._policy_switch(pred, {"traj_risk_max": 0.5, "physical_min": 0.5, "future_close_max": 0.5})
    assert switch.tolist() == [True, False, False]


def test_group_count_matches_key_multiplicity():
    keys = np.asarray(["a", "b", "a", "c", "b", "b"], dtype=object)
    counts = jlr._group_count(keys)
    assert counts.tolist() == [2.0, 3.0, 2.0, 1.0, 3.0, 3.0]
