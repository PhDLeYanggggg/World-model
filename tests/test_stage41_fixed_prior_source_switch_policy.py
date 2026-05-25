import numpy as np

from src import stage41_fixed_prior_source_switch_policy as fixed_prior


def test_xy_from_chosen_selects_expected_source():
    pack = {
        "bridge_xy": np.zeros((2, 2, 2), dtype=np.float64),
        "old_shape": {"xy": np.ones((2, 2, 2), dtype=np.float64), "shape_switch": np.asarray([True, True])},
        "gain_gate": {"xy": np.full((2, 2, 2), 2.0, dtype=np.float64), "shape_switch": np.asarray([True, False])},
    }
    xy, switch = fixed_prior._xy_from_chosen(pack, np.asarray([fixed_prior.SOURCES.index("old_shape"), fixed_prior.SOURCES.index("gain_gate")]))
    assert np.allclose(xy[0], 1.0)
    assert np.allclose(xy[1], 2.0)
    assert switch.tolist() == [True, False]


def test_fixed_prior_gain_label_uses_fixed_policy_baseline():
    labels = {
        "waypoint_xy": np.asarray([[[1.0, 0.0], [2.0, 0.0]]], dtype=np.float64),
        "waypoint_valid": np.ones((1, 2), dtype=bool),
    }
    pack = {
        "labels": labels,
        "horizon": np.asarray([50]),
        "bridge_xy": np.asarray([[[0.8, 0.0], [1.8, 0.0]]], dtype=np.float64),
        "old_shape": {"xy": np.asarray([[[1.0, 0.0], [2.0, 0.0]]], dtype=np.float64), "shape_switch": np.asarray([True])},
        "gain_gate": {"xy": np.asarray([[[0.6, 0.0], [1.6, 0.0]]], dtype=np.float64), "shape_switch": np.asarray([True])},
    }
    gain = fixed_prior._fixed_prior_gain_labels(pack, "old_shape", {"short": "bridge", "t50": "bridge", "t100": "bridge"})
    assert gain[0] > 0.0
