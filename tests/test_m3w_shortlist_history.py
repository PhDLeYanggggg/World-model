import numpy as np
import pytest

from src.world_model.m3w_shortlist_history import augment, backcast


def test_constant_motion_and_translation():
    h = np.arange(8, dtype=float)[None, :, None] * np.array([1., 2.])
    bank, floor = np.ones((1, 1, 12, 2)), np.zeros((1, 12, 2))
    i, j, scale = np.array([0]), np.array([0]), np.array([2.])
    np.testing.assert_array_equal(backcast(h, scale, bank, floor, i, j), 0.)
    h[:, -1, 0] += 3.
    a = backcast(h, scale, bank, floor, i, j)
    assert (a[:, [0, 3, 6]] > 0.).all()
    np.testing.assert_array_equal(a, backcast(h + 100, scale, bank, floor, i, j))
    with pytest.raises(ValueError):
        backcast(np.pad(h, ((0, 0), (0, 1), (0, 0))), scale, bank, floor, i, j)


def test_only_selected_action_columns_are_augmented():
    h = np.zeros((3, 8, 2))
    bank, floor = np.ones((3, 2, 12, 2)), np.zeros((3, 12, 2))
    r = np.arange(66).reshape(3, 2, 11)
    q, i, j, x = np.array([1, 2]), np.array([2, 1]), np.array([0, 1]), np.ones((2, 2))
    a, = augment([(q, i, j, x)], 'resolved', h, np.ones(3), bank, floor, r)
    for before, after in zip((q, i, j), a):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(a[3][:, :2], x)
    np.testing.assert_array_equal(a[3][:, 2:], r[i, j])
