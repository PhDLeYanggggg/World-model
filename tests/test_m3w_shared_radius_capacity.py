import itertools

import numpy as np
import pytest

from src.evaluation import m3w_shared_radius_capacity as capacity


def test_frontier_prices_missing_debit_and_allows_reallocation():
    cv = np.zeros((2, 2, 2))
    ols = cv.copy()
    ols[:, :, 0] = [[2., 1.], [0., 3.]]
    y = cv.copy()
    y[:, :, 0] = [[2., 1.], [0., np.nan]]
    valid = np.isfinite(y).all(2)
    f = capacity.radius_frontier(cv, ols, y, valid, np.ones(2, bool))
    assert capacity.upper_gain(f, 0.) == -6.
    assert capacity.upper_gain(f, .5) == -5.
    assert capacity.upper_gain(f, 2.) == -3.
    assert capacity.upper_gain(f, 5.) == 0.
    assert capacity.break_even_radius(f) == 5.
    limited = capacity.radius_frontier(cv, ols, y, valid, np.array([True, False]))
    assert capacity.break_even_radius(limited) is None


@pytest.mark.parametrize("budget", [1., 2., 4.])
def test_finite_solver_matches_exhaustive_and_keeps_cheaper_arm(budget):
    cv = np.zeros((2, 2, 2))
    y = cv.copy()
    y[:, :, 0] = [[1., 1.], [4., 4.]]
    ols = y * .5
    bank = np.stack((y * .25, y), axis=1)
    m = capacity.coefficients(cv, ols, bank, y, np.ones((2, 2), bool), 1.5)
    assert len(m["row"]) == 4
    expected = -np.inf
    for bits in itertools.product((False, True), repeat=4):
        take = np.array(bits)
        if (len(np.unique(m["row"][take])) != take.sum() or m["cost"][take].sum() > budget
                or not m["easy"][take].any()):
            continue
        if np.any(m["risk"][:, take].sum(1) > [0., 0., m["easy_cap"]]):
            continue
        gain = (m["offset"] + m["gain"][:, take].sum(1)).min()
        expected = max(expected, 100. * gain / m["cv_mass"])
    take, solver = capacity.solve(m, budget)
    assert solver["status"] == "optimal"
    assert solver["upper_pp"] == pytest.approx(expected)
    assert m["cost"][take].sum() <= budget
