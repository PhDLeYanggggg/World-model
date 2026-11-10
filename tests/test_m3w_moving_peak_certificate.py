import numpy as np
import pytest
from scipy.optimize import linprog

from reproducibility.moving_peak_certificate import verify
from scripts.audit_m3w_record_peak_break_even import minimum_peak
from scripts.package_m3w_moving_peak_certificate import aggregate
from src.evaluation.m3w_record_peak_allocation import arbitrary_bound


@pytest.mark.parametrize('budget', [0., 2., 3., 8.])
def test_dual_matches_independent_linear_program(budget):
    items = [(3., 2.), (1., 4.)]
    result = linprog([-3., -1.], A_ub=[[1., 1.]], b_ub=[budget], bounds=[(0., 2.), (0., 4.)])
    assert result.success
    assert verify.envelope(items, budget) == pytest.approx(-result.fun)


def test_aggregate_matches_original_with_censoring_and_uneven_queries():
    rng = np.random.default_rng(17)
    cv, ols, y = rng.normal(size=(3, 6, 12, 2))
    valid = rng.random(size=(6, 12)) > .2
    y[~valid] = np.nan
    query = np.array([2, 1, 3, 1, 2, 1])
    budgets = np.array([0., 2., 5.])
    row = aggregate(cv, ols, y, valid, query, budgets)
    result = verify.evaluate(row)
    original = arbitrary_bound(cv, ols, y, valid, query, budgets)
    for scope in ('query', 'record'):
        assert [result[k][scope] for k in ('cv', 'ols4')] == pytest.approx(original[scope])
    _, group, counts = np.unique(query, return_inverse=True, return_counts=True)
    cap = np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.)
    missing = np.where(valid, 0., np.linalg.norm(cv-ols, axis=-1))
    gap = row['known_ols4_advantage']
    assert result['required_peak']['available'] == pytest.approx(minimum_peak(cap, 1/counts[group], gap))
    assert result['required_peak']['charged'] == pytest.approx(
        minimum_peak(cap+missing, 1/counts[group], gap+row['missing_disagreement']))


def test_invalid_query_and_unreachable_target():
    with pytest.raises(ValueError, match='Invalid aggregate query'):
        verify.slopes([{'actors': 0, 'available': [1.] * 12}], 'available')
    assert verify.minimum_budget([(2., 1.)], 3.) is None
    assert verify.minimum_budget([(2., 1.)], 1.) == .5
