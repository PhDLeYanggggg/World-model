from itertools import product

import numpy as np
import pytest

from scripts.audit_m3w_opendd_exchange_frontier import intervals
from src.evaluation.m3w_exchange_frontier import exchange_count, solve
from src.evaluation.m3w_count_motion_capacity import respects_tokens, within_cap


@pytest.mark.parametrize('radius', [1, 2])
def test_exchange_frontier_matches_exhaustive_maximin(radius):
    n = 12
    q = np.arange(n)
    legal = np.tile([True, True, False], (n, 1))
    legal[-1, 2] = True
    anchor = np.zeros(n, dtype=int)
    anchor[[4, 9]] = 1
    move = np.tile([0., 1., 2.], (n, 1))
    move[8, 1] = 1.1
    gain = np.zeros((n, 3, 2))
    gain[:, 1] = np.column_stack((q, q+1.))
    gain[8, 1] = [80., 80.]
    gain[-1, 2] = [30., 30.]
    risk = np.zeros((n, 3, 3))
    risk[-2, 1, 1] = .03
    support = legal.copy()
    support[:, 0] = False
    offset = np.array([0., -1.])
    mass, easy_cap, cap = 100., .02, 2.

    def objective(plan):
        return float((offset + gain[q, plan].sum(0)).min())

    feasible = []
    for p in product(*(np.flatnonzero(row) for row in legal)):
        c = np.asarray(p)
        if c.sum() != anchor.sum() or exchange_count(c, anchor) > radius or not respects_tokens(c, legal[:, 1]):
            continue
        if within_cap(move[q, c].sum(), cap) and support[q, c].any() and (risk[q, c].sum(0) <= [0., 0., easy_cap]).all():
            feasible.append(objective(c))
    count, upper, status = solve((gain, risk, support, legal, offset, easy_cap, mass), move, cap, anchor, radius)
    assert status == 'bounded'
    assert objective(count) == pytest.approx(max(feasible))
    assert objective(count) <= upper <= objective(count) + 2e-6
    assert exchange_count(count, anchor) <= radius


def test_count_intervals_keep_unresolved_and_do_not_include_infeasible():
    r = lambda lo, hi, infeasible=False: dict(lower_pp=lo, upper_pp=hi, infeasible=infeasible)
    assert intervals([r(.15, .2), r(None, None), r(None, None, True), r(.02, .09)], {'floor': .1}) == {'floor': [1, 2]}
    with pytest.raises(ValueError, match='contradicts'):
        intervals([r(.15, .02)], {'floor': .1})
