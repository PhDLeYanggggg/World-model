from decimal import Decimal
import numpy as np
import pytest

from src.evaluation.m3w_stationary_label_resolution import (
    printed_interval, constant_compatible, project_supplied_h, resolution_counts,
    slice_forecast_metrics,
)


def test_printed_precision_is_explicit_and_not_a_physical_noise_bound():
    lo, hi, q = printed_interval('8.4568443e+00')
    assert q==Decimal('1e-7') and hi-lo==q
    assert constant_compatible([['1.00','2.00'], ['1.001','2.001']])
    assert not constant_compatible([['1.000','2.000'], ['1.010','2.000']])
    with pytest.raises(ValueError):
        printed_interval('NaN')


def test_projection_round_trip_and_infinity_rejection():
    h = np.array([[.1, 0, 1], [0, .2, -1], [.0001, 0, 1.]])
    pixels = np.arange(52).reshape(2, 13, 2)/3
    flat = pixels.reshape(-1, 2)
    mapped = np.c_[flat, np.ones(len(flat))] @ h.T
    xy = (mapped[:, :2]/mapped[:, 2:3]).reshape(pixels.shape)
    np.testing.assert_allclose(project_supplied_h(xy, h), pixels, atol=1e-12)
    with pytest.raises(ValueError, match='infinity'):
        project_supplied_h([[1, 0]], [[1, 0, 0], [0, 1, 0], [1, 0, 1]])


def test_resolution_keeps_group_support_separate_and_boundary_visible():
    result = resolution_counts([0, 1.0001, 3, 3], [False, True, True, True], [0, 1, 5, 5],
        [('a',), ('a',), ('b',), ('b',)], [('a',), ('a',), ('b',), ('b',)], thresholds=[1])
    assert result[0]['rows_above']==2 and result[0]['agents_above']==1
    assert result[0]['rows_within_numerical_boundary']==1
    assert result[0]['cv_error_share_above']==pytest.approx(10/11)
    assert not result[0]['labels_redefined']


def test_zero_floor_and_empty_slices_are_not_silent_successes():
    y = np.zeros((2, 12, 2)); p = np.ones_like(y)
    r = slice_forecast_metrics(p, y, np.array([True, False]))
    assert r['gain_vs_cv_pct'] is None and r['native_harm']>0
    assert not r['selection_allowed']
    assert slice_forecast_metrics(p, y, np.zeros(2, bool))['status']=='not_run_empty_slice'
