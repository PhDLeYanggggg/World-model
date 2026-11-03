import numpy as np

from src.world_model.m3w_selection_controller import labels
from src.world_model.m3w_supported_bound_targets import supported_bound_targets


def test_observed_moments_unchanged_unknowns_are_bounds_not_zero_error():
    rng = np.random.default_rng(5)
    p = rng.normal(size=(5, 7, 12, 2)); y = rng.normal(size=(5, 12, 2))
    valid = np.ones((5, 12), bool); valid[2, 5:] = False; valid[4] = False
    old, known = labels(p, y, valid, 1, 3.)
    new = supported_bound_targets(p, y, valid, 1, 3.)
    np.testing.assert_array_equal(new['moments'][known], old[known])
    np.testing.assert_array_equal(new['observed_future'], known)
    np.testing.assert_array_equal(new['unknown_bound_supervision'], ~known)
    radius = np.linalg.norm(p[4]-p[4, 1], axis=-1).max(-1)
    np.testing.assert_array_equal(new['moments'][4, :, 1], radius)
    np.testing.assert_array_equal(new['moments'][4, :, 4], radius)
    assert not new['moments'][4, :, [0, 2, 3]].any()
    assert new['target_defined'].all()


def test_unknown_or_unobserved_future_coordinates_do_not_create_targets():
    p = np.zeros((3, 7, 12, 2)); p[:, 6] = 2
    y = np.ones((3, 12, 2)); valid = np.ones((3, 12), bool)
    valid[1, 6:] = False; valid[2] = False
    a = supported_bound_targets(p, y, valid, 1, 2.)
    y[~valid] = 1e50
    b = supported_bound_targets(p, y, valid, 1, 2.)
    np.testing.assert_array_equal(a['moments'], b['moments'])
