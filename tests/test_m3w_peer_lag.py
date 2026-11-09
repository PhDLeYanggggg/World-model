import numpy as np

from src.world_model.m3w_peer_lag import candidates, peers_in_prefix, prefix_check


def histories():
    pulse = .04 * np.array([1., -1., 2., -2., 1., -1.])
    accel = np.stack((np.r_[0., pulse[:-1]], pulse, np.zeros(6)))
    speed = 1. + np.cumsum(np.column_stack((np.zeros(3), accel)), axis=1)
    h = np.zeros((3, 8, 2))
    h[:, 1:, 0] = np.cumsum(speed, axis=1)
    h[:, :, 1] = np.array([0., .7, 2.5])[:, None]
    return h, np.zeros(3, dtype=int)


def cv(h):
    return (h[:, -1:] - h[:, -2:-1]) * np.arange(1., 13.)[None, :, None]


def test_stable_lag_can_pass_the_prefix_holdout_and_add_distinct_matched_paths():
    h, q = histories()
    p = peers_in_prefix(h, q)
    assert p[0] == 1
    mask, _ = prefix_check(np.diff(h, n=2, axis=1), p)
    assert mask[0]
    paths, report = candidates(h, q, cv(h))
    assert report['nonzero_cap_rows'] > 0
    assert not np.allclose(paths[0, 0], paths[0, 1])
    radius = np.linalg.norm(paths - cv(h)[:, None], axis=-1).max(axis=-1)
    np.testing.assert_allclose(radius[:, 0], radius[:, 1], atol=1e-12)


def test_peer_choice_does_not_see_the_two_held_observed_steps():
    h, q = histories()
    altered = h.copy()
    altered[1, -2:] += 1e4
    np.testing.assert_array_equal(peers_in_prefix(h, q), peers_in_prefix(altered, q))


def test_unrelated_and_stationary_histories_do_not_make_a_lag_candidate():
    h, q = histories()
    q[0] = 1
    paths, report = candidates(h, q, cv(h))
    assert report['prefix_pass_rows'] == 0
    np.testing.assert_array_equal(paths, np.repeat(cv(h)[:, None], 2, axis=1))
    h[:] = 0.
    _, report = candidates(h, np.zeros(3), cv(h))
    assert report == dict(neighbor_rows=0, prefix_pass_rows=0, nonzero_cap_rows=0)


def test_scale_and_rotation_preserve_support_and_forecasts():
    h, q = histories()
    rot = np.array([[0., 1.], [-1., 0.]])
    shifted = 2. * h @ rot + np.array([3., 2.])
    a, sa = candidates(h, q, cv(h))
    b, sb = candidates(shifted, q, cv(shifted))
    assert sa == sb
    np.testing.assert_allclose(b, 2. * a @ rot, atol=1e-11)
