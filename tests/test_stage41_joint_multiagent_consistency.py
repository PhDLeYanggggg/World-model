import numpy as np

from src import stage41_joint_multiagent_consistency as jm


def _toy_labels():
    return {
        "current_xy": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        "cand_delta": np.asarray([[[1.0, 0.0]], [[0.0, 0.0]]], dtype=float),
        "normalizer": np.ones(2, dtype=float),
        "waypoint_xy": np.asarray(
            [
                [[0.25, 0.0], [0.50, 0.0], [0.75, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            ],
            dtype=float,
        ),
        "waypoint_valid": np.ones((2, 4), dtype=bool),
        "floor_fde": np.ones(2, dtype=float),
        "candidate_fde": np.ones((2, 1), dtype=float),
        "domain": np.asarray(["D", "D"]),
        "horizon": np.asarray([50, 50]),
        "hard": np.asarray([True, True]),
        "failure": np.asarray([True, True]),
        "easy": np.asarray([False, False]),
    }


def _toy_pred():
    return {
        "waypoint_delta": np.asarray(
            [
                [[0.25, 0.0], [0.50, 0.0], [0.75, 0.0], [1.0, 0.0]],
                [[-0.25, 0.0], [-0.50, 0.0], [-0.75, 0.0], [-1.0, 0.0]],
            ],
            dtype=float,
        ),
        "traj_risk": np.asarray([0.01, 0.01], dtype=float),
        "physical": np.asarray([1.0, 1.0], dtype=float),
        "interaction": np.asarray([0.2, 0.2], dtype=float),
        "occupancy": np.asarray([0.2, 0.2], dtype=float),
    }


def test_min_group_distance_normalized():
    xy = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
        ],
        dtype=float,
    )
    dist = jm._min_group_distance(xy, np.asarray(["g", "g"]), np.asarray([2.0, 2.0]))
    assert np.allclose(dist, [0.5, 0.5])


def test_collision_guard_reduces_unsafe_switches():
    pred = _toy_pred()
    labels = _toy_labels()
    meta = {"key": np.asarray(["g", "g"])}
    policy = {
        "slices": {
            "D|50": {
                "traj_risk_max": 1.0,
                "physical_prob_min": 0.0,
                "max_switch": 1.0,
                "easy_block": True,
                "hard_only": False,
            }
        }
    }
    base = jm._base_selection(pred, labels, policy)
    assert base["switch"].sum() == 2
    _ade, _fde, switch, diag = jm._apply_joint_variant(
        pred,
        labels,
        meta,
        policy,
        {"mode": "collision_guard", "min_sep": 0.2},
    )
    assert switch.sum() == 0
    assert diag["guarded_off"] == 2


def test_group_expand_uses_past_only_policy_signals():
    pred = _toy_pred()
    labels = _toy_labels()
    meta = {"key": np.asarray(["g", "g"])}
    policy = {"slices": {}}
    _ade, _fde, switch, diag = jm._apply_joint_variant(
        pred,
        labels,
        meta,
        policy,
        {"mode": "group_expand", "expand_risk_max": 0.02, "expand_min_sep": 0.0},
    )
    assert switch.sum() == 2
    assert diag["expanded_on"] == 2
