from itertools import product

import numpy as np
import pytest

from scripts.audit_m3w_dual_reference_capacity import evidence
from src.evaluation.m3w_dual_reference_capacity import action_masses, solve


@pytest.mark.parametrize('ols_first', [2., 4.])
def test_maximin_matches_original_scorer_exhaustively(ols_first):
    cv = np.zeros((4, 2, 2))
    y = cv.copy()
    y[:2, :, 0] = 4.
    y[3, :, 0] = 1.
    valid = np.array([[True, True], [True, False], [False, False], [True, True]])
    ols = cv.copy()
    ols[:, :, 0] = np.array([ols_first, 1., .05, .2])[:, None]
    bank = np.repeat(cv[:, None], 2, axis=1)
    bank[:, :, :, 0] = np.array([[2., 3.], [1., 2.], [.01, .05], [-.2, -.4]])[:, :, None]
    query = np.array([1, 1, 2, 2])
    inv = query-1
    counts = np.array([1, 1])
    w = np.full(4, .5)
    cut = {'easy': 5., 'hard': 8.}
    args = (cv, ols, bank, y, valid, query, 'r', cut, inv, counts, w)
    candidates = []
    for left, right, a, b in product((0, 1), (2, 3), (0, 1), (0, 1)):
        d = evidence(*args, np.array([left, right]), np.array([a, b]))
        if d['supported']:
            candidates.append(min(d['gains']))
    m = action_masses(cv, ols, bank, y, valid, w, cut['easy'])
    assert len(m['row'][m['row'] == 2]) == 2  # Unknown paths order differently by reference.
    assert not np.any((m['row'] == 0) & (m['arm'] == 0))
    x, s = solve(m, inv, counts, 3.)
    assert s['solver_status'] == 0
    got = evidence(*args, m['row'][x], m['arm'][x])
    assert got['supported']
    assert min(got['gains']) == pytest.approx(max(candidates))
    assert s['upper_gain_mass'] == pytest.approx(max(candidates))


def test_no_selected_easy_support_is_infeasible():
    cv = np.zeros((2, 1, 2))
    bank = cv[:, None]+1.
    m = action_masses(cv, cv, bank, cv, np.ones((2, 1), bool), np.full(2, .5), 2.)
    x, s = solve(m, np.zeros(2, dtype=int), np.array([1]), 3.)
    assert x is None and s['infeasible']
