from itertools import product

import numpy as np
import pytest

from src.evaluation.m3w_full_grid_count_capacity import enumerate_spend, prefix_grid_terms, solve
from src.evaluation.m3w_full_grid_risk import assess, assess_easy
from src.evaluation.m3w_streaming_menu_capacity import respects_tokens


def test_prefix_coefficients_match_direct_grid_risk_for_a_chosen_plan():
    cv = np.zeros((4, 12, 2))
    path, y = cv.copy(), cv.copy()
    path[..., 0] = np.array([.04, .25, .02, .11])[:, None]
    y[..., 0] = np.array([.1, .12, .08, .5])[:, None]
    valid = np.ones((4, 12), bool)
    valid[1, -1] = valid[2, -2] = False
    rows = np.array([[0, 1], [2, -1], [3, -1]])
    z = prefix_grid_terms(cv, path, y, valid, rows, .2)
    for counts in product(range(3), range(2), range(2)):
        take = rows[np.arange(2) < np.asarray(counts)[:, None]]
        if not len(take):
            continue
        a, b, support = z[np.arange(3), counts].sum(0)
        args = cv[take], path[take], y[take], valid[take]
        mass = np.linalg.norm(cv[take]-y[take], axis=-1)[valid[take]].sum()
        assert a == pytest.approx(mass*(assess(*args)['full_grid_upper']-.02))
        direct = assess_easy(*args, easy_cut=.2)
        assert b == pytest.approx(direct['easy_excess_supremum'])
        assert bool(support) == direct['easy_support_guaranteed']


def test_robust_optimizer_matches_enumeration_and_rejects_the_nonrobust_winner():
    legal = np.tile([True, True, False], (10, 1))
    legal[-1, 2] = True
    move = np.tile([0., 1., 0.], (10, 1))
    move[-1, 2] = 1.5
    g = np.zeros((10, 3, 2))
    g[[4, 9], 1] = [.2, .3]
    g[9, 2] = 3.
    r = np.zeros((10, 3, 3))
    grid = r.copy()
    grid[9, 1:, 2] = 1.
    grid[9, 2, 0] = 1.
    terms = (g, r, legal, np.zeros(2), 0., 1.)
    candidates = []
    for plan in product(*[np.flatnonzero(row) for row in legal]):
        ix = np.arange(10), plan
        a, b, s = grid[ix].sum(0)
        if sum(plan) == 2 and move[ix].sum() <= 2 and respects_tokens(plan, legal[:, 1]) and max(a, b) <= 0. and s:
            candidates.append(np.min(g[ix].sum(0)))
    counts, status = solve(legal, move, 2, 2., terms, grid)
    assert status == 'candidate'
    assert np.min(g[np.arange(10), counts].sum(0)) == pytest.approx(max(candidates))
    assert counts[9] == 1
    grid[..., 2] = 0.
    assert solve(legal, move, 2, 2., terms, grid) == (None, 'no_mandatory_easy_support')


def test_small_spend_enumeration_has_every_plan_and_no_illegal_edits():
    legal = np.tile([True, True, False], (11, 1))
    legal[9, 2] = True
    move = np.tile([0., 1., 0.], (11, 1))
    move[9, 2] = 2.
    expected = set()
    for plan in product(*[np.flatnonzero(row) for row in legal]):
        if sum(plan) == 2 and respects_tokens(plan, legal[:, 1]) and move[np.arange(11), plan].sum() <= 2.:
            expected.add(plan)
    assert {tuple(plan) for plan in enumerate_spend(legal, move, 2, 2.)} == expected
    assert enumerate_spend(legal, move, 2, 2., max_plans=1) is None
