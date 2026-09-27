import numpy as np
import pytest
from src.world_model.m3w_selected_risk_diagnosis import (
    signed_cost, query_sizes, support_from_fit, residual_metrics, query_residuals, exchange_accounting)


def test_signed_targets_preserve_unknown_and_easy_definition():
    q = signed_cost([1, 0, 9, np.nan], [2, 3, 4, np.nan], [3, 2, 8, np.nan], 2, 2)
    np.testing.assert_allclose(q[:3], [[.48, .48], [-.03, 0], [1.96, 0]])
    assert np.isnan(q[3]).all()


def test_signed_identity_matches_target_basis():
    from src.world_model.m3w_fixed_floor_probe import targets
    from src.world_model.m3w_fixed_floor_excess import signed
    rng = np.random.default_rng(4)
    cv, f, n = rng.uniform(0, 10, (3, 50))
    np.testing.assert_allclose(signed_cost(cv, f, n, 2, 3), signed(targets(cv, f, n, 2)[:, [2, 1, 3, 4]])/3)


def test_misaligned_cost_rejected():
    with pytest.raises(ValueError):
        signed_cost([1, np.nan], [1, 2], [1, 2], 1, 1)


def test_query_size_has_no_outcome_filter():
    np.testing.assert_array_equal(query_sizes(['a', 'a', 'b'], ['r']*3, [1]*3), [2, 2, 1])


def test_support_training_only_and_source_order_invariant():
    x = np.repeat(np.arange(8)[:, None], 6, axis=1).astype(float)
    sites = np.array(['a']*4+['b']*4)
    known = np.ones(8, bool)
    test = np.array([[3]*6, [100]*6], float)
    a = support_from_fit(x, sites, known, test, np.zeros(6), np.ones(6))
    b = support_from_fit(x[::-1], sites[::-1], known, test, np.zeros(6), np.ones(6))
    np.testing.assert_array_equal(a['distance'], b['distance'])
    assert a['thresholds'] == b['thresholds']
    assert a['outside_both'].tolist() == [False, True]
    # Evaluation distribution must not influence fitting radii.
    c = support_from_fit(x, sites, known, test*100, np.zeros(6), np.ones(6))
    assert c['thresholds'] == a['thresholds']


def test_support_unknown_fitting_rows_do_not_enter_tree():
    x = np.zeros((4, 6)); x[2:] = 1; x[1] = 100
    known = np.array([True, False, True, True])
    a = support_from_fit(x, np.array(['a', 'a', 'b', 'b']), known, x[1:2], np.zeros(6), np.ones(6))
    assert a['outside_both'][0]


def test_residual_empty_is_undefined_not_zero():
    r = residual_metrics(np.zeros((2, 2)), np.ones((2, 2)), np.zeros(2, bool),
                         np.zeros((2, 3), bool), np.zeros(2, bool), np.ones(2))
    assert r['all_optimism_mean'] is None and r['controller_proxy_fraction'] is None


def test_residual_unknowns_only_affect_causal_coverage():
    r = residual_metrics(np.zeros((2, 2)), np.array([[1, 2], [np.nan, np.nan]]), np.ones(2, bool),
                         np.array([[True, True, False], [False, False, True]]), np.array([False, True]), np.ones(2))
    assert r['all_optimism_mean'] == 1 and r['known'] == r['unknown'] == 1
    assert r['controller_proxy_fraction'] == .5 and r['controller_proxy_known_fraction'] == 1


def test_query_risk_does_not_certify_partial_labels():
    p = np.array([[-1., 0], [-2, 0], [-1, 0]])
    y = np.array([[2., 0], [np.nan, np.nan], [3, 0]])
    r = query_residuals(p, y, np.ones(3, bool), ['r']*3, [1, 1, 2])
    assert r['complete_queries'] == 1 and r['unknown_selected_queries'] == 1
    assert r['all_predicted_safe_actual_excess_queries'] == 1


def test_exchange_separates_lost_benefit_from_added_harm():
    r = exchange_accounting([10, 10, 10, np.nan], [5, 13, 9, np.nan],
                            np.array([True, False, True, False]), np.array([False, True, True, True]))
    assert r['removed_benefit'] == 5 and r['added_harm'] == 3 and r['gain_change'] == -8
    assert r['added_unknown'] == 1 and r['selected_before'] == 2 and r['selected_after'] == 3


def test_exchange_known_zero_denominator_undefined():
    r = exchange_accounting([0], [1], np.array([False]), np.array([True]))
    assert r['gain_change'] == -1 and r['gain_change_over_floor_pp'] is None


def test_exchange_random_accounting():
    rng = np.random.default_rng(14)
    for _ in range(25):
        f, n = rng.random((2, 50))
        a, b = rng.random((2, 50)) < .3
        r = exchange_accounting(f, n, a, b)
        assert r['added_count']-r['removed_count'] == b.sum()-a.sum()


def test_query_report_json_serializable():
    import json
    json.dumps(query_residuals(np.zeros((2, 2)), np.ones((2, 2)),
                               np.ones(2, bool), ['r']*2, [1, 2]), allow_nan=False)


def test_locality_reduce_does_not_hide_undefined_views():
    from scripts.diagnose_m3w_selected_risk import locality_reduce
    rows = [dict(site='a', metric={'x':1}), dict(site='a', metric={'x':None}), dict(site='b', metric={'x':3})]
    r = locality_reduce(rows, 'x', ['a', 'b'], True)
    assert r['mean'] == 2 and r['undefined_views'] == 1 and r['ci95'] is None


def test_locality_reduce_averages_views_before_bootstrap():
    from scripts.diagnose_m3w_selected_risk import locality_reduce
    rows = [dict(site='a', metric={'x':1})]*5+[dict(site='b', metric={'x':3})]
    r = locality_reduce(rows, 'x', ['a', 'b'], True)
    assert r['mean'] == 2 and r['ci95'] == [1, 3]
