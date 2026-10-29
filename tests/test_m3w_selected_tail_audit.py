import numpy as np
import pytest

from src.evaluation import m3w_selected_tail_audit as a


def test_accounting_separates_denominator_from_harm():
    p = np.array([[2., .01, 10., 10., .01], [2., .1, 10., 10., .1]])
    y = np.array([[0., 1., 1., 1., 1.], [0., .1, .1, .1, .1]])
    c = a.accounting(p, y, np.ones(2, bool), 2.)
    assert c['easy']['violating_rows'] == 2
    assert c['easy']['violating_rows_still_allowed_with_true_reference'] == 1
    assert c['easy']['violating_rows_still_allowed_with_true_harm'] == 1
    np.testing.assert_allclose(c['easy']['observed_excess'], (.98+.098)/2)


def test_unknown_labels_do_not_change_action_or_bin():
    p = np.array([[1., .01, 2., 2., .01]]*3)
    y = np.array([[0., .3, 1., 1., .2]]*3)
    eligible = np.ones(3, bool)
    before = a.actions(p, eligible)
    y[1] = np.nan
    assert a.accounting(p, y, before, 1.)['unknown'] == 1
    np.testing.assert_array_equal(before, a.actions(p, eligible))
    np.testing.assert_array_equal(a.bin_ids([0., 1., 5.], [1., 2., 3.]), [0, 1, 3])


def test_bin_edges_are_fit_only_and_keep_duplicate_bins():
    fit = np.array([1., 1., 1.])
    e = a.fitting_bins(fit, np.ones(3, bool))
    assert e == [1., 1., 1.]
    assert a.bin_ids([0., 1., 1000.], e).tolist() == [0, 3, 3]
    assert e == a.fitting_bins(fit, np.ones(3, bool))
    with pytest.raises(ValueError):
        a.fitting_bins(fit, np.zeros(3, bool))


def test_harm_supremum_matches_scalar_odds_and_zero_envelope():
    p = np.array([[.2, .1, 1., .2, .02], [0., 0., 1., 1., 0.]])
    limit = a.harm_upper(p, np.array([1., 0.]))
    h = .8*(.1/.7*np.exp(5))/(1+.1/.7*np.exp(5))
    q = (.2/.8*np.exp(5))/(1+.2/.8*np.exp(5))
    np.testing.assert_allclose(limit[0], [h, h*q])
    assert limit[1].tolist() == [0., 0.]


def test_all_controls_and_unknowns_in_accounting():
    p = np.array([[1., .01, 2., 2., .01]]*4)
    y = np.array([[0., .2, 1., 1., .1]]*4)
    y[3] = np.nan
    r = a.view({k: p for k in a.ARMS}, y, np.ones(4)*2, np.ones(4, bool),
               np.arange(4), [1., 2., 3.], 1.)
    a.validate_view(r)
    assert r['arms']['isolated_risk']['cohorts']['own']['unknown'] == 1
    r['arms']['isolated_risk']['own_radius_bins'][0]['rows'] += 1
    with pytest.raises(ValueError):
        a.validate_view(r)


def test_radius_unclipped_and_uses_frozen_preprocessing():
    x = np.array([[1., 2.], [11., 2.]])
    np.testing.assert_allclose(a.radius(x, dict(mean=[1., 2.], std=[2., 1.])), [0., np.sqrt(12.5)])
