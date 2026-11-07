import numpy as np
import pytest

from src.evaluation.m3w_pair_completion_witness import completion_ledger, extreme


def test_completion_preserves_observations_and_shares_overlapping_targets():
    agents = np.array([19, 19])
    frames = np.array([7, 8])
    h = np.zeros((2, 8, 2))
    h[:, :, 0] = frames[:, None]+np.arange(-7, 1)
    target = np.full((2, 3, 2), np.nan)
    valid = np.array([[True, False, False], [False, False, True]])
    target[0, 0] = [8., 0.]
    target[1, 2] = [11., 0.]
    ix, seen, base, radius, gaps = completion_ledger(agents, frames, h, target, valid)
    assert ix[0, 1] == ix[1, 0]
    np.testing.assert_allclose(base[ix[0], 0], [8., 9., 10.])
    np.testing.assert_array_equal(radius[ix[0]], [0., 1., 1.])
    assert gaps[ix[0, 1]] == gaps[ix[0, 2]]
    assert seen[ix[0, 0]]
    target[0, 0] = [7., 0.]
    with pytest.raises(ValueError, match='conflict'):
        completion_ledger(agents, frames, h, target, valid)


def test_constructive_worlds_can_have_opposite_signs_without_moving_known_points():
    x = np.array([[0., 0.], [1., 0.], [1., 0.]])
    radius = np.array([0., 1., 1.])
    gap = np.array([-1, 1, 1])
    edges = np.array([[1, 0], [2, 0]])
    left = np.array([[0., 0.], [0., 0.]])
    right = np.array([[2., 0.], [2., 0.]])
    weight = np.array([.5, .5])
    lo, y_lo, _ = extreme(x, radius, gap, edges, left, right, weight, -1.)
    hi, y_hi, _ = extreme(x, radius, gap, edges, left, right, weight, 1.)
    assert lo == pytest.approx(-2.)
    assert hi == pytest.approx(2.)
    np.testing.assert_array_equal(y_lo[0], x[0])
    np.testing.assert_array_equal(y_hi[0], x[0])
    np.testing.assert_array_equal(y_hi[1], y_hi[2])
