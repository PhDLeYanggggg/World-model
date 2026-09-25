import numpy as np
import pytest
from scripts.diagnose_m3w_european_fixed_producer_roles import switch_accounting


def test_added_harm_and_removed_benefit_both_count_as_losses():
    r = switch_accounting([4., 1., 1., 4.], [2., 2., 2., 2.],
        np.array([1, 0, 1, 0], bool), np.array([0, 1, 0, 1], bool))
    assert r['native_pixel_terms'] == dict(added_harm=.5, added_benefit=.25, removed_lost_benefit=.25, removed_avoided_harm=.5)
    assert r['mean_change'] == 0. and r['added_rows'] == r['removed_rows'] == 2


def test_identical_policy_has_zero_change_not_zero_error():
    b = np.array([True, False]); r = switch_accounting([1., 4.], [2., 2.], b, b)
    assert r['new_error_mean'] == 1.5 and r['mean_change'] == 0.


def test_missing_or_zero_reference_not_epsilon_percentage():
    with pytest.raises(ValueError): switch_accounting([np.nan], [1.], np.array([True]), np.array([False]))
    r = switch_accounting([2.], [0.], np.array([True]), np.array([False]))
    assert r['net_degradation_percent'] is None and r['mean_change'] == 2.
