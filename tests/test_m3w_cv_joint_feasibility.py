import itertools

import numpy as np

from src.evaluation import m3w_cv_joint_feasibility as audit
from src.evaluation.m3w_cv_anchor_budget import candidate_terms
from src.evaluation.m3w_selection_forecast_readout import policy
from scripts.audit_m3w_cv_joint_matched_shape import rematch


def test_joint_optimum_matches_original_policy_and_exhaustive_search():
    cv = np.zeros((4, 3, 2))
    y = np.broadcast_to(np.array([.2, .3, 1., .6])[:, None, None], cv.shape)
    ols = .32 * y
    bank = np.stack((ols, .85*y, -.2*y), axis=1)
    query = np.array([0, 0, 1, 1])
    valid = np.ones((4, 3), bool)
    valid[2, -1] = False
    w = np.full(4, .5)
    d = audit.coefficients(cv, ols, bank, y, valid, query, .65)
    assert len(d['row']) < d['actions_before_prune']
    for cap in (1, 2):
        vals = []
        for acts in itertools.product(range(-1, 3), repeat=4):
            acts = np.array(acts)
            use = acts >= 0
            if np.bincount(query[use], minlength=2).max() > cap:
                continue
            path = cv.copy()
            path[use] = bank[use, acts[use]]
            p = policy(path, cv, cv, y, valid, np.repeat('r', 4), easy_cut=.65, hard_cut=1.)
            if not p['recordings']['r']['supported']:
                continue
            m, b, g, _ = candidate_terms(cv, ols, path[:, None], y, valid)
            vals.append(min(w @ g[:, 0, 0], w @ (b + g[:, 0, 1])) / (w @ m) * 100.)
        take, result = audit.solve(d, cap)
        assert take is not None
        np.testing.assert_allclose((d['offset'] + d['gain'][take].sum(0)).min(), max(vals), atol=1e-7)
        assert result['upper_pp'] >= max(vals) - 1e-7


def test_no_easy_is_not_a_pass():
    cv = np.zeros((2, 3, 2))
    y = np.ones_like(cv)
    d = audit.coefficients(cv, y, y[:, None], y, np.ones((2, 3), bool), np.arange(2), .1)
    take, result = audit.solve(d, 1)
    assert take is None and result['status'] == 'undefined_support'


def test_exact_shape_rematch_can_recover_with_a_scaled_analytic_direction():
    cv = np.zeros((3, 4, 2))
    y = np.broadcast_to(np.array([.2, .5, .8])[:, None, None], cv.shape)
    ols = .25 * y
    valid = np.ones((3, 4), bool)
    analytic = np.stack((ols, -.2*y), axis=1)
    original = y[:, None]
    assert rematch(cv, ols, analytic, original, [[0, 0], [2, 0]], y, valid,
                   np.array([0, 1, 1]), .5, 1., 1) == 'verified_analytic_rematch'
