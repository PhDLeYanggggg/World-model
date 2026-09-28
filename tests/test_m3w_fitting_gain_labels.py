import numpy as np
import pytest

from src.world_model.m3w_fitting_gain_labels import gain_labels, verify_alignment


def test_signed_gain_keeps_benefit_harm_tie_and_missing_distinct():
    cv = np.array([1., 1., 0., 5., np.nan])
    floor = np.array([2., 2., 0., 3., np.nan])
    neural = np.array([1., 4., 0., 2., np.nan])
    out = gain_labels(cv, floor, neural, easy_cut=2., scale=2.)
    np.testing.assert_equal(out['signed_gain'], [.5, -1., 0., .5, np.nan])
    np.testing.assert_equal(out['positive_benefit'], [.5, 0., 0., .5, np.nan])
    np.testing.assert_equal(out['positive_harm'], [0., 1., 0., 0., np.nan])
    np.testing.assert_equal(out['easy'], [1., 1., 0., 0., np.nan])
    np.testing.assert_equal(out['known'], [True, True, True, True, False])
    np.testing.assert_equal(out['signed_gain'], out['positive_benefit']-out['positive_harm'])


def test_scaling_costs_and_scale_does_not_change_labels():
    args = [np.array([1., 3.]), np.array([2., 4.]), np.array([3., 2.])]
    a = gain_labels(*args, easy_cut=2., scale=2.)
    b = gain_labels(*(v*7 for v in args), easy_cut=14., scale=14.)
    for key in a: np.testing.assert_equal(a[key], b[key])


@pytest.mark.parametrize('bad', ['mask', 'negative', 'scale', 'cut', 'shape', 'inf'])
def test_rejects_invalid_costs(bad):
    a, b, c = (np.array([1., 2.]) for _ in range(3)); scale, cut = 1., 2.
    if bad == 'mask': b[0] = np.nan
    if bad == 'negative': c[0] = -1
    if bad == 'scale': scale = 0
    if bad == 'cut': cut = np.nan
    if bad == 'shape': c = c[:, None]
    if bad == 'inf': a[0] = b[0] = c[0] = np.inf
    with pytest.raises(ValueError): gain_labels(a, b, c, easy_cut=cut, scale=scale)


def test_all_missing_remains_unknown_not_zero():
    out = gain_labels([np.nan], [np.nan], [np.nan], easy_cut=1., scale=1.)
    assert not out['known'][0]
    assert all(np.isnan(v[0]) for k, v in out.items() if k != 'known')


def test_alignment_checks_existing_targets_and_source_exclusions():
    y = gain_labels([1., 2.], [2., 3.], [1., 4.], easy_cut=1., scale=2.)
    old = np.array([[1., 1., 0.], [0., 1.5, .5]])
    roles = dict(training_sites=['a', 'b'], held_sites=['c'], producer_sites=['d'], controller_sites=['e'])
    verify_alignment(np.array([2, 7]), ['a', 'b'], y, old, roles)
    with pytest.raises(ValueError): verify_alignment(np.array([2, 2]), ['a', 'b'], y, old, roles)
    with pytest.raises(ValueError): verify_alignment(np.array([7, 2]), ['a', 'b'], y, old, roles)
    with pytest.raises(ValueError): verify_alignment(np.array([2, 7]), ['a', 'c'], y, old, roles)
    wrong = old.copy(); wrong[0, 2] = .5
    with pytest.raises(AssertionError): verify_alignment(np.array([2, 7]), ['a', 'b'], y, wrong, roles)
