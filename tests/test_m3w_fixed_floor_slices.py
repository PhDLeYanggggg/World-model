import numpy as np
import pytest

from src.world_model.m3w_fixed_floor_slices import (
    causal_axes, source_weights, fit_edges, assign_bins, evaluate_slice, reduce_slices)


def geometry(n=4):
    g = np.zeros((n, 476))
    g[:, :16] = np.stack((np.arange(-7, 1), np.zeros(8)), -1).ravel()
    g[:, 16:24] = np.arange(-7, 1)
    return g


def test_axes_use_only_causal_geometry_and_ignore_invalid_neighbor_padding():
    g = geometry()
    b = np.zeros((4, 12, 2)); n = b + 1
    x = np.zeros((4, 380)); pr = dict(mean=x[0], std=x[0]+1, clip=10)
    a = causal_axes(g, b, n, x, pr, 2.)
    g[:, 38:166] = 9999
    changed = causal_axes(g, b, n, x, pr, 2.)
    for key in a: np.testing.assert_array_equal(a[key], changed[key])
    np.testing.assert_allclose(a['path_nonlinearity'], 0)
    np.testing.assert_allclose(a['mean_turn_radians'], 0)
    np.testing.assert_allclose(a['neighbor_occupancy'], 0)
    g[:, 16] = 1
    with pytest.raises(ValueError): causal_axes(g, b, n, x, pr, 2.)


def test_fitting_edges_balance_sources_and_do_not_consult_held_values():
    s = np.array(['a']*100 + ['b'])
    w = source_weights(s)
    assert np.isclose(w[:100].sum(), .5) and w[-1] == .5
    edge = fit_edges(np.r_[np.zeros(100), 100], w)
    np.testing.assert_array_equal(edge, [0, 100])
    np.testing.assert_array_equal(assign_bins([-10, 1, 100, 1000], edge), [0, 1, 2, 2])
    np.testing.assert_array_equal(assign_bins([0, 1], [0, 0]), [2, 2])
    with pytest.raises(ValueError): fit_edges([np.nan, 1], [.5, .5])


def test_future_unknown_rows_stay_in_decisions_but_not_loss_or_outcome():
    r = evaluate_slice(np.array([True, True, True]), np.array([1., 1., np.nan]),
        np.array([.5, 2., np.nan]), np.array([True, False, True]),
        np.array([True, True, True]), np.array([-.01, .02, -3.]), 1.)
    assert r['rows'] == 3 and r['known'] == 2 and r['unknown_selected'] == 1
    assert r['selected'] == 2 and r['selected_known'] == 1
    assert r['harm_sum'] == 0 and r['benefit_sum'] == .5
    assert r['eligible_oracle_benefit_sum'] == .5
    assert r['sq_error_sum'] == pytest.approx((.01)**2 + (.02-.98)**2)
    assert r['floor_sum'] == 2


def test_empty_selected_reference_is_undefined_not_zero_risk():
    row = evaluate_slice(np.ones(2, bool), np.array([1., 2.]), np.array([0., 3.]),
        np.zeros(2, bool), np.ones(2, bool), np.zeros(2), 1.)
    rows = [dict(site=s, role='held', policy='excess', axis='all', bin='all', sums=row)
            for s in ('a', 'b')]
    d = reduce_slices(rows, ['a', 'b'], 200, 7)['held']['excess']['all/all']
    assert d['harm_percent']['point'] is None
    assert d['net_gain_percent']['point'] == 0
    assert d['known_percent']['point'] == 100


def test_missing_fixed_roster_site_is_not_dropped():
    row = evaluate_slice(np.ones(1, bool), np.array([1.]), np.array([.5]),
        np.ones(1, bool), np.ones(1, bool), np.array([-.02]), 1.)
    rows = [dict(site='a', role='held', policy='excess', axis='all', bin='all', sums=row)]
    d = reduce_slices(rows, ['a', 'b'], 200, 7)['held']['excess']['all/all']
    assert d['net_gain_percent']['point'] is None
    assert d['net_gain_percent']['by_site']['a'] == 50
    assert d['net_gain_percent']['by_site']['b'] is None


def test_slices_partition_additive_counts_and_normalized_errors():
    f = np.array([1., 2., 3., np.nan]); n = np.array([2., 1., 4., np.nan])
    take = np.ones(4, bool); elig = take.copy(); q = np.zeros(4)
    all_rows = evaluate_slice(take, f, n, take, elig, q, 2.)
    parts = [evaluate_slice(np.arange(4)%2 == i, f, n, take, elig, q, 2.) for i in range(2)]
    for key in all_rows: assert parts[0][key]+parts[1][key] == pytest.approx(all_rows[key])
    with pytest.raises(ValueError): evaluate_slice(take, f, n, take, ~elig, q, 2.)
