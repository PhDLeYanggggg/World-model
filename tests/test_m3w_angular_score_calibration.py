import numpy as np
import pytest

from src.evaluation.m3w_angular_score_calibration import MARGINS, indices, select_threshold


def row(gain=0., dual=2, rotations=0):
    return dict(gain_pp=gain, dual=dual, rotated_edits=rotations)


def test_stable_calibration_cannot_accept_loss_at_any_donor():
    a, b = [[row() for _ in MARGINS] for _ in range(2)]
    a[0], b[0] = row(.4, rotations=10), row(-.1, rotations=10)
    a[1], b[1] = row(.1, rotations=6), row(.05, rotations=4)
    assert select_threshold([a, b]) == 0.
    assert select_threshold([a, b], stable=True) == .001


def test_thresholds_without_original_dual_support_fall_back():
    a, b = [[row() for _ in MARGINS] for _ in range(2)]
    a[0], b[0] = row(10., dual=1, rotations=3), row(10., rotations=3)
    assert select_threshold([a, b]) is None
    with pytest.raises(ValueError):
        select_threshold([a, b[:-1]])


def test_margin_is_strict_and_keep_is_exact():
    idx = np.array([0, 1, 2])
    gain = np.array([.01, .02, -.1])
    np.testing.assert_array_equal(indices(idx, gain, .01), [0, 2, 0])
    np.testing.assert_array_equal(indices(idx, gain, None), [0, 0, 0])
