import numpy as np
import pytest
from src.evaluation.m3w_partial_neighbor_refit import masks, slice_metrics, paired_localities, endpoint_references


def test_training_cut_partition_and_missing_are_not_easy():
    m = masks(np.array([0., .5, 2., 5., np.nan]), 1., 4.)
    np.testing.assert_array_equal(m['positive_easy'], [False, True, False, False, False])
    assert m['zero_CV'].sum() == 1 and m['hard'].sum() == 1
    with pytest.raises(ValueError): masks(np.ones(3), 2, 1)


def test_gains_harm_and_zero_references_remain_honest():
    a, b = np.array([1., 3., np.nan]), np.array([2., 2., np.nan])
    m = slice_metrics(a, b, b, b, np.ones(3, bool))
    assert m['gain_vs_legacy_percent'] == 0 and m['rows'] == 2
    assert m['mean_positive_harm_vs_legacy'] == .5
    assert m['mean_positive_gain_vs_legacy'] == .5
    z = slice_metrics(a[:2], np.zeros(2), np.zeros(2), np.zeros(2), np.ones(2, bool))
    assert z['gain_vs_legacy_percent'] is None and z['absolute_harm_vs_legacy'] == 2
    with pytest.raises(ValueError): slice_metrics(a, np.zeros(3), b, b, np.ones(3, bool))


def test_context_replicas_are_not_independent_bootstrap_units():
    rows = [dict(site=s, metric={'gain':v}) for s,v in [('a', 10), ('b', -20)]]
    a = paired_localities(rows, ['a', 'b'], 'gain')
    b = paired_localities(rows*5, ['a', 'b'], 'gain')
    assert a == b and a['point'] == -5 and a['resampled_localities'] == 2
    assert paired_localities(rows, ['a', 'b', 'c'], 'gain')['point'] is None


def test_undefined_guard_is_not_silently_dropped():
    r = [dict(site='a', metric={'gain':10}), dict(site='a', metric={'gain':None})]
    assert paired_localities(r, ['a'], 'gain')['ci95'] is None


def test_fde_comparison_uses_fde_reference_not_ade():
    data = dict(baseline_ade=np.array([[1., 2.], [3., 4.]]),
                baseline_fde=np.array([[10., 20.], [30., 40.]]))
    reference, cv = endpoint_references(data, np.array([1, 0]), 0, 'FDE')
    np.testing.assert_array_equal(reference, [30., 10.])
    np.testing.assert_array_equal(cv, [40., 20.])
    result = slice_metrics(cv/2, cv, reference, cv, np.ones(2, bool))
    assert result['gain_vs_CV_percent'] == 50
    with pytest.raises(ValueError): endpoint_references(data, np.array([0]), 0, 'unknown')
