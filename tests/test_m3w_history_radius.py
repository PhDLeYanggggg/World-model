import numpy as np

from src.world_model import m3w_history_radius as m


def test_constant_past_motion_does_not_produce_backtest_error():
    h = np.arange(8.)[None, :, None]*np.array([[[1., 2.]], [[3., 2.]]])
    scale, bucket = m.describe(h)
    assert (scale > 1.).all() and (bucket == 0).all()


def test_clip_retains_direction_and_identity_and_respects_row_radius():
    cv = np.zeros((2, 12, 2))
    bank = np.zeros((2, 2, 12, 2))
    bank[0, 0, :, 0] = 10.
    bank[1, 1, :, 1] = .1
    new = m.clip(cv, bank, np.array([.25, 1.]))
    np.testing.assert_allclose(new[0, 0, :, 0], .25)
    np.testing.assert_array_equal(new[1], bank[1])
    np.testing.assert_array_equal(new[0, 1], bank[0, 1])


def test_lower_reference_is_fitted_without_the_held_source():
    sources = np.array(['a', 'b', 'held'])
    q, values = m.fit(np.array([1., 4., 0.]), np.ones(3, bool),
        np.ones(3), np.zeros(3), sources, np.ones(3), 'held')
    assert q == 1. and (values == 1.).all()
