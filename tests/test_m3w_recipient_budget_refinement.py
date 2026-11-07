import numpy as np

from scripts.refine_m3w_recipient_budget_capacity import prune
from src.evaluation.m3w_recipient_budget_capacity import solve


def test_unknown_dominance_preserves_better_cost_tradeoffs_and_optimum():
    cv = np.zeros((4, 12, 2))
    ols = cv.copy()
    ols[2:, :, 0] = .5
    y = cv.copy()
    y[:2, :, 0] = np.array([10., 20.])[:, None]
    valid = np.ones((4, 12), bool)
    valid[2:] = False
    bank = np.zeros((4, 5, 12, 2))
    bank[:2, 0] = y[:2]
    bank[2:, :, :, 0] = np.array([0., .125, .25, .375, .625])[None, :, None]
    mask = prune(cv, ols, bank, valid)
    np.testing.assert_array_equal(mask[2:], [[False, True, True, True, False]]*2)
    outcomes = [solve(cv, ols, bank, y, valid, np.array([0, 4]), np.array([2]), m,
                      'case', 15., 19.) for m in (np.ones_like(mask), mask)]
    assert all(r['optimum_proved'] and r['feasible'] for r in outcomes)
    np.testing.assert_allclose(outcomes[0]['witness_mass'], outcomes[1]['witness_mass'])
