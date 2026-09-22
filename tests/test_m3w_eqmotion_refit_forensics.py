import numpy as np
import pytest
from scripts.audit_m3w_eqmotion_cost_refit import groups, summarize


def test_groups_depend_only_on_frozen_choices():
    g = groups(np.array([True, True, False]), np.array([False, True, True]))
    np.testing.assert_array_equal(g['frozen_only'], [True, False, False])
    np.testing.assert_array_equal(g['refit_only'], [False, False, True])
    np.testing.assert_array_equal(g['overlap'], [False, True, False])
    with pytest.raises(ValueError): groups(np.ones(3), np.ones(3))


def test_unknown_and_partial_are_not_zero_cost_and_both_heads_use_same_rows():
    mask = np.ones(3, bool); valid = np.array([[True, True], [True, False], [False, False]])
    cv, candidate = np.array([5., 100., np.nan]), np.array([7., 1., np.nan])
    old, new = np.array([[0., 1.], [9., 9.], [8., 8.]]), np.array([[0., 2.], [3., 3.], [4., 4.]])
    r = summarize(mask, valid, cv, candidate, old, new, np.ones(3), np.ones(3))
    assert r['complete'] == 1 and r['incomplete'] == 2 and r['unknown_ADE'] == 1
    assert r['costs']['realized_harm'] == 2 and r['costs']['refit']['native_cost_MSE'] == 0
    assert r['costs']['frozen']['native_cost_MSE'] == .5
    mask[0] = False
    assert summarize(mask, valid, cv, candidate, old, new, np.ones(3), np.ones(3))['costs'] is None
