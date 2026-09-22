import numpy as np
import pytest
from src.evaluation.m3w_risk_ranking import risk_fraction, fixed_count, choices, overlap, MATCHED


def test_zero_benefit_and_empty_cost_are_last_without_epsilon():
    p = np.array([[0., 0.], [0, 9], [2, 0], [2, 1], [1e300, 1e300]])
    np.testing.assert_array_equal(risk_fraction(p), [1, 1, 0, 1/3, .5])


def test_same_count_risk_and_gain_distinguish_large_risky_forecast():
    p = np.array([[100., 50.], [5, .1], [2, .01], [4, 2]])
    e = np.ones(4, bool); ids = np.arange(4)
    np.testing.assert_array_equal(fixed_count(p, e, ids, 2, 'ratio'), [False, True, True, False])
    np.testing.assert_array_equal(fixed_count(p, e, ids, 2, 'gain'), [True, True, False, False])


def test_risk_ranking_invariant_to_common_positive_harm_multiplier():
    p = np.array([[3., .1], [7, 1], [9, 4], [2, 2], [0, 0]])
    e = np.ones(5, bool); ids = np.arange(5)
    a = fixed_count(p, e, ids, 3, 'ratio')
    for factor in (.001, 7, 100):
        q = p.copy(); q[:, 1] *= factor
        np.testing.assert_array_equal(fixed_count(q, e, ids, 3, 'ratio'), a)


def test_ties_use_row_id_not_row_position():
    p = np.ones((4, 2)); ids = np.array([8, 2, 9, 1]); e = np.ones(4, bool)
    for rank in ('ratio', 'gain'):
        np.testing.assert_array_equal(fixed_count(p, e, ids, 2, rank), [False, True, False, True])
        assert not fixed_count(p, e, ids, 0, rank).any()


def test_choices_recover_forest_and_same_count_without_future_arguments():
    rng = np.random.default_rng(81)
    f, nn = rng.uniform(0, 10, (2, 200, 2))
    h = rng.normal(size=(200, 8, 2)); h[:7, -1] = h[:7, -2]
    d = np.ones(200); d[7:12] = 0; ids = rng.permutation(200)
    eligible = (d > 0) & np.any(h[:, -1] != h[:, -2], axis=1)
    strict = lambda p: eligible & (p[:, 0] > p[:, 1]) & (p[:, 1] <= .1*p[:, 0])
    out = choices(f, nn, h, d, ids, strict(f), strict(nn))
    for name in MATCHED:
        assert out[name].sum() == strict(f).sum()
        assert not out[name][:12].any()
    np.testing.assert_array_equal(out['forest_ratio'], strict(f))


@pytest.mark.parametrize('fault', ['negative', 'nan', 'duplicate', 'count', 'mask', 'kind'])
def test_invalid_controls_fail_closed(fault):
    p = np.ones((4, 2)); ids = np.arange(4); e = np.ones(4, bool); k = 2; rule = 'ratio'
    if fault == 'negative': p[0, 1] = -1
    if fault == 'nan': p[0, 0] = np.nan
    if fault == 'duplicate': ids[0] = ids[1]
    if fault == 'count': k = 5
    if fault == 'mask': e = e.astype(int)
    if fault == 'kind': rule = 'oracle'
    with pytest.raises(ValueError): fixed_count(p, e, ids, k, rule)


def test_frozen_choice_change_rejected():
    p = np.array([[4., .1]]*3); h = np.zeros((3, 8, 2)); h[:, -1] = 1
    with pytest.raises(ValueError, match='Frozen'):
        choices(p, p, h, np.ones(3), np.arange(3), np.zeros(3, bool), np.ones(3, bool))


def test_overlap_handles_zero_and_disjoint_choices():
    assert overlap(np.zeros(3, bool), np.zeros(3, bool))['jaccard'] is None
    r = overlap(np.array([True, False, True]), np.array([False, True, True]))
    assert r['common'] == 1 and r['a_only'] == 1 and r['jaccard'] == 1/3
