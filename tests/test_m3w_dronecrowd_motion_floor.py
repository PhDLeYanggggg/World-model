import numpy as np
import pytest

from src.world_model import m3w_dronecrowd_motion_floor as model


def test_all_methods_reproduce_a_linear_prefix():
    t = np.arange(8)
    h = np.stack((np.column_stack((t, -t)), np.column_stack((2*t+100, .5*t))))
    y = model.paths(model.coefficients(h))
    expect = np.diff(h[:, -2:], axis=1)*np.arange(1, 13)[None, :, None]
    np.testing.assert_allclose(y, np.broadcast_to(expect[:, None], y.shape))


def test_peer_origin_shift_changes_individual_paths_not_pair_geometry():
    h = np.zeros((3, 8, 2))
    h[:, :, 0] = np.arange(8)
    h[:, -1, 0] += [3., 3., 50.]
    y = model.paths(model.coefficients(h))
    delta = y[:, 4]-y[:, 1]
    assert abs(delta[0, 0, 0]) > 0.
    np.testing.assert_allclose(delta, np.broadcast_to(delta[0, 0], delta.shape))
    np.testing.assert_allclose(y[:, 4]-y[:1, 4], y[:, 1]-y[:1, 1])


def test_observed_prefix_controls_use_relative_positions():
    h = np.random.default_rng(13).normal(size=(5, 8, 2))
    shift = np.array([20., -31.])
    np.testing.assert_allclose(model.coefficients(h+shift), model.coefficients(h), atol=1e-12)
    with pytest.raises(ValueError, match="eight-step"):
        model.coefficients(h[:, :-1])


def test_guard_reverts_if_cv_is_already_perfect_on_every_probe():
    h = np.zeros((1, 8, 2))
    h[:, :, 0] = np.arange(8)
    h[:, -1, 0] = 20.
    c = model.coefficients(h)
    np.testing.assert_array_equal(c[:, -1], c[:, 0])
