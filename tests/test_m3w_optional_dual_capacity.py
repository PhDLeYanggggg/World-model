from itertools import product
import numpy as np

from scripts.audit_m3w_clipped_reference_bridge import clipped_bank
from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_optional_dual_capacity import coefficients, solve
from src.evaluation.m3w_selection_forecast_readout import policy


def test_optional_maximin_matches_enumeration_under_partial_future():
    cv = np.zeros((3, 2, 2))
    ols = cv.copy()
    ols[..., 0] = [[.4, .8], [1., 1.], [.8, 1.6]]
    y = cv.copy()
    y[..., 0] = [[.5, 1.], [2., 2.], [1., np.nan]]
    valid = np.isfinite(y).all(2)
    other = cv.copy()
    other[..., 0] = [[.6, 1.], [2., 2.], [1., 0.]]
    bank = np.stack((ols, other), axis=1)
    m = coefficients(cv, ols, bank, y, valid, 1.)
    take, status = solve(m)
    best = -np.inf
    for choices in product(range(3), repeat=3):
        p = np.stack([cv[j] if k == 0 else bank[j, k-1] for j, k in enumerate(choices)])
        if policy(p, cv, cv, y, valid, ["r"]*3, easy_cut=1., hard_cut=3.)["recordings"]["r"]["supported"]:
            best = max(best, min(paired(ref, p, y, valid)[1].sum() for ref in (cv, ols)))
    assert status["status"] == "solved" and take is not None
    np.testing.assert_allclose(status["upper_pp"], 100*best/m["cv_mass"])


def test_zero_switch_is_not_easy_support():
    cv = np.zeros((1, 2, 2))
    bank = cv[:, None]
    y = np.ones_like(cv)
    assert solve(coefficients(cv, cv, bank, y, np.ones((1, 2), bool), 2.))[1]["status"] == "no_legal_easy"


def test_clipped_reference_preserves_the_cap_and_original_support():
    cv = np.zeros((3, 2, 2))
    ols = np.ones_like(cv)
    b = clipped_bank(cv, ols, np.array([0., .5, 2.]))
    assert not b[0].any()
    np.testing.assert_allclose(np.linalg.norm(b[1, -1], axis=1), .5)
    np.testing.assert_array_equal(b[2, -1], ols[2])
