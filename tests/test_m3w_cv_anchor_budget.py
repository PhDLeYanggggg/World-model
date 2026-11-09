import itertools

import numpy as np

from src.evaluation import m3w_cv_anchor_budget as audit
from src.evaluation.m3w_selection_forecast_readout import policy


def test_top_k_exact_and_optional():
    q = np.array([8, 1, 8, 8, 1])
    score = np.array([[3., 2.], [-4., -2.], [2., 1.], [-1., 1.], [5., 6.]])
    order, offsets = audit.row_ranks(q)
    curve = audit.cap_curve(score.max(1), order, offsets)
    np.testing.assert_array_equal(curve, [0., 9., 11., 12.])
    for k, opt in enumerate(curve):
        options = itertools.product(range(-1, 2), repeat=len(q))
        best = max(sum(score[i, c] for i, c in enumerate(x) if c >= 0) for x in options
                   if all(sum(c >= 0 for c, qq in zip(x, q) if qq == v) <= k for v in set(q)))
        choice = audit.choice_at_cap(score, order, offsets, k)
        assert best == opt == sum(score[i, c] for i, c in enumerate(choice) if c >= 0)


def test_arbitrary_optimum_reached_per_reference_with_partial_labels():
    rng = np.random.default_rng(4)
    cv, ols, target = rng.normal(size=(3, 6, 2, 2))
    valid = rng.uniform(size=(6, 2)) > .45
    target[~valid] = np.nan
    bank = np.stack((cv, np.where(valid[..., None], target, cv), np.where(valid[..., None], target, ols)), axis=1)
    _, base, actions, arbitrary = audit.candidate_terms(cv, ols, bank, target, valid)
    np.testing.assert_array_equal(actions[:, 0], 0.)
    np.testing.assert_allclose(arbitrary, actions.max(1))
    assert np.isfinite(base).all()


def test_risk_excess_reconstructs_original_before_support_and_signed_easy():
    cv = np.zeros((3, 2, 2))
    y = np.ones_like(cv)
    y[1, 0] = 2.
    valid = np.array([[1, 1], [1, 0], [0, 0]], bool)
    p = cv - .5
    original = policy(p, cv, cv, y, valid, np.array(['r']*3), easy_cut=2., hard_cut=4.)['recordings']['r']
    excess = audit.risk_excess(cv, p[:, None], y, valid, 2.).sum(0)[0]
    denom = np.sqrt(2.) + np.sqrt(8.)
    np.testing.assert_allclose(excess, [denom*(original['completion_all_risk_upper']-.02),
                                        np.sqrt(2.)*(original['completion_easy_risk_upper']-.02)])
