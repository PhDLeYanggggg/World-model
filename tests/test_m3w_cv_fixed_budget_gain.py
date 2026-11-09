import numpy as np

from src.evaluation.m3w_cv_anchor_budget import candidate_terms
from src.world_model.m3w_cv_fixed_budget_gain import Budget, targets


def test_masked_decomposition_matches_paired_increment():
    rng = np.random.default_rng(5)
    cv, ols, path, y = rng.normal(size=(4, 5, 12, 2))
    valid = rng.random((5, 12)) > .5
    valid[0] = True
    valid[1] = False
    y[~valid] = np.nan
    scale = np.arange(1, 6)
    geom, missing, observed, gain = targets(cv, ols, path, y, valid, scale)
    _, _, ref, _ = candidate_terms(cv, ols, path[:, None], y, valid)
    np.testing.assert_allclose(gain, ref[:, 0]/(12*scale[:, None]), atol=1e-7)
    np.testing.assert_allclose(gain, observed[:, None]+(missing[..., None]*geom).mean(1), atol=1e-7)


def test_budget_keeps_count_risk_reference_and_gain():
    legal = np.array([[True, True], [True, False], [True, True]])
    risk = np.array([[[1, 1, -1], [0, 0, 0]], [[1, 0, -1], [0, 0, 0]],
                     [[.5, .5, -2], [3, 1, -1]]])
    ref = np.array([[2., 2.], [1., 1.], [2., 2.]])
    gain = np.array([[2., 2.], [1., 1.], [2., 2.]])
    anchor = np.array([0, -1, -1])
    b = Budget(legal, risk, ref, gain, anchor)
    np.testing.assert_array_equal(b.solve([[0, 9], [100, 0], [8, 100]]), [-1, -1, 0])
    np.testing.assert_array_equal(b.solve([[0, -1], [100, 0], [-1, 100]]), anchor)
    empty = np.full(3, -1)
    np.testing.assert_array_equal(Budget(legal, risk, ref, gain, empty).solve(np.ones((3, 2))), empty)
