import itertools
import numpy as np
import pytest

from src.world_model.m3w_easy_component_diagnostic import (
    compose, weights, objective, attribution, diagnose,
)


def fixture():
    sites = np.array(['a', 'a', 'a', 'b'])
    recordings = np.array(['r', 'r', 'r', 's'])
    frames = np.array([1, 1, 2, 1])
    truth = np.array([[1., 1., .1], [0., 2., .2], [1., 3., 0.], [1., 2., .3]])
    left = np.array([[.7, 1.1, .08], [.3, 2.1, .1], [.8, 2.5, .02], [.6, 2.4, .2]])
    right = left + np.array([.01, .02, .03])
    return left, right, truth, sites, recordings, frames


def test_source_and_query_balanced_weights():
    _, _, y, s, r, f = fixture()
    w, q = weights(s, r, f, np.ones(len(y), bool))
    np.testing.assert_allclose(w, [.125, .125, .25, .5])
    assert q[0] == q[1] and len(set(q)) == 3
    assert w.sum() == 1


def test_unknown_rows_are_excluded_not_zero_targets():
    a, b, y, s, r, f = fixture()
    y[0] = np.nan
    doc = diagnose(a, b, y, s, r, f)
    assert doc['unknown_rows_excluded'] == 1 and doc['known_rows'] == 3
    assert doc['sources']['a']['arms']['uncapped']['partitions']['all']['rows'] == 2


def test_objective_matches_explicit_source_query_average():
    _, _, _, s, r, f = fixture()
    w, q = weights(s, r, f, np.ones(4, bool))
    e = np.array([1., -1., 2., 3.])
    expected_rows = .5*(.5*(1+4)+9)
    expected_queries = .5*(.5*(0+4)+9)
    assert objective(e, w, q)['marginal'] == .5*(expected_rows+expected_queries)


def test_complete_error_decomposition_and_partition_additivity():
    a, b, y, s, r, f = fixture()
    doc = diagnose(a, b, y, s, r, f)
    for group in doc['sources'].values():
        for row in group['arms'].values():
            p = row['partitions']
            assert sum(p[k]['weight_mass'] for k in ('not_easy', 'easy_zero_harm', 'easy_positive_harm')) == pytest.approx(1)
            assert sum(p[k]['MSE_contribution'] for k in ('not_easy', 'easy_zero_harm', 'easy_positive_harm')) == pytest.approx(p['all']['MSE_contribution'])
            for stats in p.values():
                assert sum(stats['squared_components'])+sum(stats['cross_components']) == pytest.approx(stats['MSE_contribution'])


def test_permutation_attribution_matches_set_formula_and_efficiency():
    vals = {str(i): float(i*i-3*i) for i in range(8)}
    got = attribution(vals)
    expected = []
    for bit in range(3):
        other = [i for i in range(3) if i != bit]
        v = (vals[str(1 << bit)]-vals['0'])/3
        v += sum((vals[str((1 << j) | (1 << bit))]-vals[str(1 << j)])/6 for j in other)
        mask = sum(1 << j for j in other)
        v += (vals['7']-vals[str(mask)])/3
        expected.append(v)
    np.testing.assert_allclose(got, expected)
    assert sum(got) == pytest.approx(vals['7']-vals['0'])


def test_unchanged_components_have_zero_attribution():
    a, _, y, s, r, f = fixture()
    d = diagnose(a, a.copy(), y, s, r, f)
    for row in d['sources'].values():
        np.testing.assert_array_equal(row['attribution']['marginal'], np.zeros(3))
        np.testing.assert_array_equal(row['attribution']['row_MSE'], np.zeros(3))


def test_probability_changes_magnitude_not_positive_probability_sign():
    x = np.array([[.1, 2., .1], [.9, 2., .1]])
    assert np.sign(compose(x)[0]) == np.sign(compose(x)[1])
    assert compose(x)[1] == pytest.approx(9*compose(x)[0])


def test_no_easy_labels_keep_conditional_quantities_undefined():
    a, b, y, s, r, f = fixture(); y[:, 0] = 0
    d = diagnose(a, b, y, s, r, f)
    for row in d['sources'].values():
        for arm in row['arms'].values():
            assert arm['conditional_reference_MSE'] is None
            assert arm['partitions']['easy_positive_harm']['conditional_MSE'] is None


@pytest.mark.parametrize('bad', ['probability', 'negative_cost', 'nan_prediction', 'partial_label', 'nonbinary_easy', 'length'])
def test_invalid_inputs_rejected(bad):
    a, b, y, s, r, f = fixture()
    if bad == 'probability': a[0, 0] = 1.1
    if bad == 'negative_cost': a[0, 1] = -1
    if bad == 'nan_prediction': a[0, 0] = np.nan
    if bad == 'partial_label': y[0, 1] = np.nan
    if bad == 'nonbinary_easy': y[0, 0] = .5
    if bad == 'length': f = f[:-1]
    with pytest.raises(ValueError): diagnose(a, b, y, s, r, f)


def test_all_unknown_rejected():
    a, b, y, s, r, f = fixture(); y[:] = np.nan
    with pytest.raises(ValueError): diagnose(a, b, y, s, r, f)
