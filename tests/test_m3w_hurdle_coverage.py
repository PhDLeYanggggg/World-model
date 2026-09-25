import inspect

import numpy as np
import pytest

from src.evaluation.m3w_hurdle_coverage import matched_choices, decompose


def fixture():
    ids = np.array([9, 7, 5, 3, 1, 0])
    sites = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    product = np.column_stack((np.ones(6), [.01, .03, .04, .01, .015, .05]))
    hurdle = np.column_stack((np.ones(6), [.04, .015, .01, .04, .03, .01]))
    return np.ones(6), product, hurdle, np.ones(6, bool), sites, ids


def test_two_anchors_match_counts_per_locality_not_only_globally():
    choices, records = matched_choices(*fixture(), budget=.02)
    sites = fixture()[4]
    for anchor in ('product', 'hurdle'):
        for site in ('a', 'b'):
            m = sites == site
            assert choices[anchor + '_original'][m].sum() == choices[
                ('hurdle_at_product' if anchor == 'product' else 'product_at_hurdle')][m].sum()
    assert choices['product_original'].sum() == choices['hurdle_original'].sum() == 3
    assert records['a']['product_count'] != records['a']['hurdle_count']
    assert records['b']['product_count'] != records['b']['hurdle_count']


def test_tie_breaking_and_permutation_equivariance():
    args = list(fixture())
    args[2][:, 1] = .03
    choices, _ = matched_choices(*args, budget=.02)
    assert set(args[5][choices['hurdle_at_product']]) == {5, 0, 1}
    permutation = np.array([4, 2, 5, 0, 3, 1])
    reordered, _ = matched_choices(*(x[permutation] for x in args), budget=.02)
    for key in choices:
        np.testing.assert_array_equal(choices[key][permutation], reordered[key])


def test_zero_and_full_coverage_and_eligibility():
    args = list(fixture())
    args[1][:, 1] = 0
    args[2][:, 1] = .03
    args[0][0] = 0
    args[3][1] = False
    choices, _ = matched_choices(*args, budget=.02)
    np.testing.assert_array_equal(choices['hurdle_at_product'], choices['product_original'])
    assert not choices['hurdle_original'].any()
    assert not choices['product_at_hurdle'].any()
    assert all(not x[:2].any() for x in choices.values())


def test_no_future_signature_and_invalid_scores_fail_closed():
    assert set(inspect.signature(matched_choices).parameters) == {
        'utility', 'product', 'hurdle', 'moving', 'sites', 'ids', 'budget'}
    args = list(fixture())
    args[1][0, 1] = np.nan
    with pytest.raises(ValueError):
        matched_choices(*args, budget=.02)
    args = list(fixture())
    args[2][0, 0] = 0
    with pytest.raises(ValueError, match='common causal support'):
        matched_choices(*args, budget=.02)


def test_decomposition_uses_common_reference_and_retains_unknowns():
    cv = np.array([2., 4., np.nan, 3., 6.])
    sites = np.array(['a', 'a', 'a', 'b', 'b'])
    errors = {
        'product_original': cv.copy(),
        'hurdle_at_product': cv * .9,
        'product_at_hurdle': cv * .95,
        'hurdle_original': cv * .8,
    }
    out = decompose(errors, cv, sites, mask=np.ones(5, bool), resamples=3000, seed=1)
    assert out['unknown_rows'] == 1
    assert out['total']['mean_gain_difference_pp'] == pytest.approx(20)
    assert out['ranking_at_product_count']['mean_gain_difference_pp'] == pytest.approx(10)
    assert out['coverage_with_hurdle_ranking']['mean_gain_difference_pp'] == pytest.approx(10)
    assert out['coverage_with_product_ranking']['mean_gain_difference_pp'] == pytest.approx(5)
    assert out['ranking_at_hurdle_count']['mean_gain_difference_pp'] == pytest.approx(15)
    for row in out['by_locality'].values():
        assert row['total'] == pytest.approx(row['ranking_at_product_count'] + row['coverage_with_hurdle_ranking'])
        assert row['total'] == pytest.approx(row['ranking_at_hurdle_count'] + row['coverage_with_product_ranking'])


def test_zero_reference_or_empty_locality_is_undefined_not_dropped():
    cv = np.array([0., 0., 2., 4.])
    sites = np.array(['a', 'a', 'b', 'b'])
    errors = {key: cv.copy() for key in (
        'product_original', 'hurdle_at_product', 'product_at_hurdle', 'hurdle_original')}
    out = decompose(errors, cv, sites, mask=np.ones(4, bool), resamples=10, seed=1)
    assert out['total'] is None
    assert out['by_locality']['a']['status'] == 'zero_reference_percentage_undefined'
    errors['hurdle_original'][0] = np.nan
    with pytest.raises(ValueError):
        decompose(errors, cv, sites, mask=np.ones(4, bool), resamples=10, seed=1)
