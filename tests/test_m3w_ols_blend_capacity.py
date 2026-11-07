import numpy as np

from src.evaluation.m3w_ols_blend_capacity import add_blends, prune_unknown


def test_original_bank_prefix_and_no_near_identity_added():
    cv = np.zeros((2, 2, 2))
    ols = cv+.5
    p = np.stack((cv-.5, cv+1.), axis=1)
    out = add_blends(cv, ols, p, [0, 1])
    np.testing.assert_array_equal(out[:, :2], p)
    np.testing.assert_array_equal(out[:, 3], cv)  # Exact midpoint is identity.
    np.testing.assert_allclose(out[:, 2], cv+.25)
    np.testing.assert_allclose(out[:, 7], cv+.875)


def test_unknown_dominance_keeps_legal_tradeoffs_even_if_abstention_is_better():
    cv = np.zeros((2, 2, 2))
    ols = cv+2.
    bank = np.stack((cv-1., cv-2., cv, cv+1., cv+2., cv+3., cv+1.), axis=1)
    # Only first two non-CV options for row 1, both worse than abstention vs OLS.
    bank[1, 3:] = cv[1]
    out = prune_unknown(cv, ols, bank, np.zeros((2, 2), bool))
    assert np.flatnonzero((out[0] != cv[0]).any(axis=(1, 2))).tolist() == [3, 4]
    assert np.flatnonzero((out[1] != cv[1]).any(axis=(1, 2))).tolist() == [0]
    np.testing.assert_array_equal(prune_unknown(cv, ols, bank, np.ones((2, 2), bool)), bank)
