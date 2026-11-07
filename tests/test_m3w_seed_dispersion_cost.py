import numpy as np

from src.world_model import m3w_seed_dispersion_cost as method


def test_one_seed_features_do_not_change_when_other_seeds_change():
    h = np.zeros((2, 8, 2)); p = np.zeros((3, 2, 12, 2))
    a, da, floor, scale = method.inputs(h, p, 0, np.ones(2))
    p[1, :, :, 0] = 3
    b, db, _, _ = method.inputs(h+4, p, 0, np.ones(2))
    np.testing.assert_array_equal(a, b)
    assert not da.any() and (db > 0).all()
    assert not floor.any() and (scale == 1).all()


def test_targets_mask_future_and_do_not_impute_unknown_rows():
    floor = np.zeros((3, 12, 2)); candidate = floor.copy(); candidate[:, :, 0] = 2.
    future = np.full_like(floor, np.nan); future[:2, 0] = [[2, 0], [0, 0]]
    mask = np.isfinite(future).all(-1)
    cost = method.targets(floor, candidate, future, mask, np.ones(3))
    np.testing.assert_array_equal(cost[:2], [[2, 0], [-2, 2]])
    assert np.isnan(cost[2]).all()


def test_count_and_tie_breaking_are_fixed_per_query():
    query = np.array([4, 4, 4, 4, 2, 2, 3]); ids = np.array([14, 12, 11, 13, 3, 2, 9])
    predictions = np.array([[2, 2], [2, 0], [2, 0], [2, 0], [3, 3], [1, 1], [100, 0]])
    np.testing.assert_array_equal(method.matched_choices(query, ids, predictions),
        [False, True, True, False, False, True, False])
