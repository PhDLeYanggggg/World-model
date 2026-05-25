import numpy as np

from src import stage41_pairwise_shape_switch_policy as pairwise


def _tiny_pack():
    labels = {
        "waypoint_xy": np.asarray(
            [
                [[1.0, 0.0], [2.0, 0.0]],
                [[1.0, 0.0], [2.0, 0.0]],
                [[1.0, 0.0], [2.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        "horizon": np.asarray([50, 50, 100], dtype=np.int16),
        "hard": np.asarray([False, True, False]),
        "failure": np.asarray([False, False, True]),
        "normalizer": np.ones(3, dtype=np.float64),
        "waypoint_valid": np.ones((3, 2), dtype=bool),
    }
    bridge_xy = np.asarray(
        [
            [[0.8, 0.0], [1.8, 0.0]],
            [[0.7, 0.0], [1.7, 0.0]],
            [[0.6, 0.0], [1.6, 0.0]],
        ],
        dtype=np.float64,
    )
    old_xy = np.asarray(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[0.4, 0.0], [1.4, 0.0]],
            [[0.7, 0.0], [1.7, 0.0]],
        ],
        dtype=np.float64,
    )
    gain_xy = bridge_xy.copy()
    return {
        "labels": labels,
        "bridge_xy": bridge_xy,
        "old_shape": {"xy": old_xy, "shape_switch": np.asarray([True, True, True])},
        "gain_gate": {"xy": gain_xy, "shape_switch": np.asarray([True, True, True])},
        "horizon": labels["horizon"],
    }


def test_gain_labels_are_relative_to_bridge():
    pack = _tiny_pack()
    gain = pairwise._gain_labels(pack, "old_shape")
    assert gain[0] > 0.0
    assert gain[1] < 0.0


def test_pairwise_selection_falls_back_when_gain_too_small():
    pack = _tiny_pack()
    pred = {
        "old_shape": {"gain": np.asarray([0.1, 0.1, 0.1]), "harm": np.zeros(3)},
        "gain_gate": {"gain": np.asarray([0.0, 0.0, 0.0]), "harm": np.zeros(3)},
    }
    policy = {
        "harm_weight": 1.0,
        "gain_min": 1.0,
        "score_min": 0.0,
        "margin_min": 0.0,
        "harm_max": 1.0,
        "max_rate_h10": 0.05,
        "max_rate_h25": 0.05,
        "max_rate_h50": 0.05,
        "max_rate_h100": 0.05,
    }
    _xy, switch, chosen = pairwise._choose_pairwise_sources(pack, pred, policy)
    assert np.all(chosen == pairwise.SOURCES.index("bridge"))
    assert not np.any(switch)


def test_pairwise_selection_respects_horizon_rate_limit():
    pack = _tiny_pack()
    pred = {
        "old_shape": {"gain": np.asarray([3.0, 2.0, 1.0]), "harm": np.zeros(3)},
        "gain_gate": {"gain": np.asarray([0.0, 0.0, 0.0]), "harm": np.zeros(3)},
    }
    policy = {
        "harm_weight": 0.0,
        "gain_min": 0.0,
        "score_min": 0.0,
        "margin_min": 0.0,
        "harm_max": 1.0,
        "max_rate_h10": 0.0,
        "max_rate_h25": 0.0,
        "max_rate_h50": 0.5,
        "max_rate_h100": 0.0,
    }
    _xy, switch, chosen = pairwise._choose_pairwise_sources(pack, pred, policy)
    assert int(np.sum(chosen == pairwise.SOURCES.index("old_shape"))) == 1
    assert int(np.sum(switch)) == 1
