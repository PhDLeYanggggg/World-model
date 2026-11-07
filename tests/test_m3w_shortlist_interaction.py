import numpy as np
import pytest

from src.world_model.m3w_shortlist_interaction import OBSERVED, augment, features


def test_only_within_query_and_candidate_specific_proximity():
    h = np.broadcast_to(np.array([[0., 0.], [3., 0.], [0., 0.]])[:, None], (3, 8, 2))
    f = np.zeros((3, 12, 2))
    b = np.stack((f, f.copy()), axis=1)
    b[0, 1, :, 0] = 3.
    s, q = np.ones(3), [np.array([0, 1]), np.array([2])]
    x = features(h, s, b, f, q)
    np.testing.assert_array_equal(x, features(h + 10., s, b, f, q))
    np.testing.assert_array_equal(x[2], 0.)
    np.testing.assert_array_equal(x[:, 0, OBSERVED+2:], 0.)
    np.testing.assert_array_equal(x[:, 0, :OBSERVED], x[:, 1, :OBSERVED])
    assert (x[0, 1, OBSERVED+2:] > 0.).all()
    assert not x[1, 1, OBSERVED+2:].any()
    with pytest.raises(ValueError):
        features(h[:, :7], s, b, f, q)


def test_reorder_rows_and_ablate_rollout():
    rng = np.random.default_rng(7)
    h, f = rng.normal(size=(3, 8, 2)), rng.normal(size=(3, 12, 2))
    b = rng.normal(size=(3, 2, 12, 2))
    rows = np.arange(3)
    x = features(h, np.ones(3), b, f, [rows])
    y = features(h[::-1], np.ones(3), b[::-1], f[::-1], [rows])
    np.testing.assert_allclose(x, y[::-1])
    i, j, base = rows[:2], np.array([0, 1]), np.ones((2, 2))
    a, = augment([(rows, i, j, base)], x, rollout=False)
    np.testing.assert_array_equal(a[3][:, 2:], x[i, j, :OBSERVED])
