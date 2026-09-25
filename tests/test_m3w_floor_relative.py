import numpy as np
import pytest
from src.world_model.m3w_floor_relative import relative_targets, matched_features, assert_producer_exclusion, fixed_rank_scale


def test_easy_membership_stays_cv_defined_not_new_floor_defined():
    cv = np.array([1., 10., 0., np.nan])
    floor = np.array([12., .1, 2., np.nan]); candidate = np.array([13., 9., 4., np.nan])
    utility, risk = relative_targets(cv, floor, candidate, reference='floor', event='easy', easy_cut=2)
    np.testing.assert_array_equal(utility[:3], [[0, 1], [0, 8.9], [0, 2]])
    np.testing.assert_array_equal(risk[:3], [[12, 1], [0, 0], [0, 0]])
    assert np.isnan(risk[3]).all()


def test_cost_reference_not_input_label():
    cv = np.array([10., 2.]); d = np.array([8., 3.]); n = np.array([9., 1.])
    cu, cr = relative_targets(cv, d, n, reference='cv', event='all', easy_cut=2)
    du, dr = relative_targets(cv, d, n, reference='floor', event='all', easy_cut=2)
    np.testing.assert_array_equal(cu, [[1, 0], [1, 0]])
    np.testing.assert_array_equal(du, [[0, 1], [2, 0]])
    np.testing.assert_array_equal(cr, [[10, 0], [2, 0]])
    np.testing.assert_array_equal(dr, [[8, 1], [3, 0]])


@pytest.mark.parametrize('field', ['fit_target_overlap', 'fit_outer_overlap', 'outside_parent', 'incomplete_halves'])
def test_producer_exclusion_is_full_chain(field):
    train, target, outer, parent = {'a', 'b'}, {'c', 'd'}, {'e'}, {'a', 'b', 'c', 'd'}
    if field == 'fit_target_overlap': target.add('a')
    if field == 'fit_outer_overlap': train.add('e')
    if field == 'outside_parent': train = {'a', 'z'}
    if field == 'incomplete_halves': target = {'c'}
    with pytest.raises(ValueError):
        assert_producer_exclusion(train, target, outer, parent)


def test_two_halves_exclude_all_outer_sources():
    assert_producer_exclusion({'a', 'b'}, {'c', 'd'}, {'e', 'f'}, {'a', 'b', 'c', 'd'})


def test_matched_features_use_rollouts_and_past_only():
    g = np.zeros((2, 476), np.float32)
    g[:, 16:24] = np.arange(-7, 1)/12
    b = np.ones((2, 12, 2)); d = b.copy(); d[1] *= 2; n = b*3
    x, envelope = matched_features(g, b, d, n, np.array([False, True]))
    assert x.shape == (2, 380)
    np.testing.assert_array_equal(x[:, -1], [0, 1])
    np.testing.assert_allclose(envelope, 2*np.sqrt(2))
    y, _ = matched_features(g, b, d, n, np.array([False, True]))
    np.testing.assert_array_equal(x, y)


def test_malformed_targets_rejected():
    with pytest.raises(ValueError):
        relative_targets(np.array([np.nan]), np.array([1.]), np.array([2.]), reference='floor', event='all', easy_cut=2)


def test_fixed_scale_uses_cloned_sampler_and_only_supplied_fitting_arrays():
    import torch
    target = np.array([[1., 2.], [2., .2], [3., .1], [1., 1.]])
    sites = np.array(['a', 'a', 'b', 'b'])
    pr = dict(known=np.ones(4, bool), cost_scale=1.)
    before = torch.get_rng_state().clone()
    a = fixed_rank_scale(target, sites, pr, seed=17, batch_size=16, batches=4)
    b = fixed_rank_scale(target, sites, pr, seed=17, batch_size=16, batches=4)
    assert a == b and a > 0
    assert torch.equal(before, torch.get_rng_state())
