import numpy as np

from src.evaluation.m3w_recipient_budget_capacity import solve


def test_recombining_legal_pairs_cannot_exceed_original_per_query_cap():
    cv = np.zeros((3, 12, 2))
    y = cv.copy()
    y[:, :, 0] = np.array([1., 3., 6.])[:, None]
    bank = np.stack((y, .5*y, -y), axis=1)
    out = solve(cv, cv, bank, y, np.ones((3, 12), bool), np.array([0, 3]),
                np.array([2]), np.ones((3, 3), bool), 'fixture', 1.5, 4.)
    assert out['feasible'] and out['optimum_proved'] and out['switches'] == 2
    np.testing.assert_allclose(out['witness_mass'], (1.+6.)/3)
    assert out['decisions'] == 3


def test_unknown_rows_cannot_buy_easy_support_and_identity_is_not_an_action():
    cv = np.zeros((3, 12, 2))
    y = cv.copy()
    y[0, :, 0] = 1.
    valid = np.ones((3, 12), bool)
    valid[2] = False
    proposal = cv.copy()
    proposal[2, :, 0] = 2.
    out = solve(cv, proposal, proposal[:, None], y, valid, np.array([0, 3]),
                np.array([2]), np.ones((3, 1), bool), 'fixture', 1.5, 4.)
    assert out['proved_infeasible'] and not out['feasible']


def test_unknown_paths_keep_their_paired_ols_disagreement():
    cv = np.zeros((3, 12, 2))
    y = cv.copy()
    y[0, :, 0] = 10.
    valid = np.ones((3, 12), bool)
    valid[2] = False
    ols = cv.copy()
    ols[2, :, 0] = .15
    bank = ols[:, None].copy()
    bank[0, 0, :, 0] = 10.
    out = solve(cv, ols, bank, y, valid, np.array([0, 3]), np.array([2]),
                np.ones((3, 1), bool), 'fixture', 15., 20.)
    assert out['feasible'] and out['unknown_switches'] == 1
    np.testing.assert_allclose(out['witness_mass'], 10./3.)
