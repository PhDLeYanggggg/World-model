import numpy as np
import pytest
from src.evaluation.m3w_cost_support_audit import fit_cuts, strata, cost_stats


def test_quantiles_ignore_unsupported_training_rows_and_partition_ties():
    values = np.array([0., 1., 2., 2., 100000.]); known = np.array([True, True, True, True, False])
    cuts = fit_cuts(values, values, known)
    assert cuts['disagreement'] == [2., 2., 2.]
    out = strata(values, cuts['disagreement'])
    np.testing.assert_array_equal(sum(out.values()), np.ones(5, int))
    assert out['above_q99'][-1] and out['zero'][0]


def test_no_positive_training_support_is_not_silently_inferred():
    cuts = fit_cuts(np.zeros(3), np.zeros(3), np.ones(3, bool))
    assert cuts['disagreement'] is None
    masks = strata(np.array([0., 3.]), cuts['disagreement'])
    np.testing.assert_array_equal(masks['positive_no_training_support'], [False, True])


def test_unknowns_remain_excluded_and_constant_is_train_supplied():
    y = np.array([[2., 0.], [0., 2.], [np.nan, np.nan]])
    p = np.array([[1., 0.], [0., 1.], [100., 100.]])
    r = cost_stats(p, y, np.array([2., 2., 2.]), np.ones(3, bool), np.array([True, True, False]), [1., 1.], [.5, .5])
    assert r['complete_rows'] == 2 and r['excluded_incomplete'] == 1
    assert r['statistics']['native_MSE'] == .5
    assert r['statistics']['fraction_MSE'] == .125
    assert r['statistics']['native_constant_MSE'] == 1
    assert r['statistics']['net_spearman'] == pytest.approx(1.)


def test_empty_slice_and_zero_disagreement():
    r = cost_stats(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2), np.zeros(2, bool), np.ones(2, bool), [1, 1], [.3, .2])
    assert r['statistics'] is None
    r = cost_stats(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2), np.ones(2, bool), np.ones(2, bool), [1, 1], [.3, .2])
    assert r['statistics']['native_constant_MSE'] == 0 and r['statistics']['net_spearman'] is None


@pytest.mark.parametrize('value', [np.nan, -1., np.inf])
def test_invalid_causal_support_rejected(value):
    with pytest.raises(ValueError): strata(np.array([value]), [1., 2., 3.])
