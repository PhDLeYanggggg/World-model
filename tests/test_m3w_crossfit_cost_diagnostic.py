import numpy as np
import pytest

from src.evaluation.m3w_crossfit_cost_diagnostic import decompose_costs


def test_site_weighted_attribution_and_seedwise_oracle():
    cv = np.array([0., 2., 0., 4.])
    errors = np.array([[1., 1., 2., 5.], [1., 3., 2., 3.]])
    result = decompose_costs(cv, errors, np.array(['a', 'a', 'b', 'b']), np.ones(4))
    assert result['equal_site_excess_total_pp'] == pytest.approx(50)
    assert result['equal_site_excess_decomposition_pp'] == pytest.approx({'zero_target': 50, 'nonzero_target': 0})
    assert result['equal_site_oracle_gain_percent'] == pytest.approx(100 / 6)
    assert result['subsets']['zero_target']['gain_percent'] is None
    assert result['subsets']['nonzero_target']['gain_percent'] == 0


def test_no_easy_rows_is_explicit_and_invalid_costs_rejected():
    result = decompose_costs(np.array([2.]), np.array([[1.]]), np.array(['a']), np.ones(1))
    assert result['easy_rows'] == 0
    assert result['subsets']['zero_target']['native_pixel_mean_excess'] is None
    with pytest.raises(ValueError):
        decompose_costs(np.array([0.]), np.array([[1.]]), np.array(['a']), np.ones(1))
    with pytest.raises(ValueError):
        decompose_costs(np.array([2.]), np.array([[np.nan]]), np.array(['a']), np.ones(1))
