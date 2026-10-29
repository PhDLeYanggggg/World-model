import numpy as np
import pytest
from src.evaluation.m3w_reference_harm_readout import harm_summary, compare_arms


def test_net_benefit_does_not_cancel_positive_harm():
    r = harm_summary(np.array([0., 1.1]), np.array([1., 1.]))
    assert r['known_net_degradation_percent'] < 0
    assert r['known_positive_harm_percent'] == pytest.approx(5.)
    assert r['known_budget_pass'] is False


def test_empty_zero_unknown_never_become_safety_pass():
    for m,r in (([], []), ([0.], [0.]), ([np.nan], [np.nan])):
        result = harm_summary(np.asarray(m), np.asarray(r))
        assert result['support_status'] == 'not_estimable'
        assert result['known_budget_pass'] is None
        assert result['complete_population_budget_claim'] is False
    z = harm_summary(np.array([1.]), np.array([0.]))
    assert z['zero_reference_harmed_rows'] == 1


def test_mismatched_support_rejected():
    with pytest.raises(ValueError):
        harm_summary(np.array([1.]), np.array([np.nan]))


def test_cluster_effect_does_not_treat_rows_as_independent():
    sites = np.array(['a', 'b', 'b', 'b'])
    old, new, ref = np.ones(4), np.array([.5, .9, .9, .9]), np.ones(4)
    result = compare_arms(new, old, ref, sites, ['a','b'])
    assert result['equal_locality_gain_difference_pp'] == pytest.approx(30.)
    assert result['paired_locality_bootstrap_ci95_pp'] == pytest.approx([10.,50.])


def test_missing_required_locality_not_silently_dropped():
    x = np.array([1.])
    r = compare_arms(x, x, x, np.array(['a']), ['a','b'])
    assert r['equal_locality_gain_difference_pp'] is None
    assert r['paired_locality_bootstrap_ci95_pp'] is None
