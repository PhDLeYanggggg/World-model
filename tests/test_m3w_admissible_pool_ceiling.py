import numpy as np

from src.evaluation.m3w_admissible_pool_ceiling import solve


def test_full_bank_can_rescue_an_infeasible_prescreened_pool():
    cv = np.zeros((3, 12, 2)); y = cv + .1
    p = np.stack((cv + .3, y), axis=1)
    raw = np.array([0, -1, -1]); q = np.ones(3); valid = np.ones((3, 12), bool)
    bad = np.tile([True, False], (3, 1)); full = np.ones_like(bad)
    a = solve(cv, p, bad, raw, y, valid, q, 'r', 1., 2., fixed=True)
    b = solve(cv, p, full, raw, y, valid, q, 'r', 1., 2., fixed=True)
    assert a['status'] == 'infeasible'
    assert b['status'] == 'optimal' and b['switches'] == 1 and b['risk']['supported']


def test_empty_matched_policy_does_not_supply_reference():
    cv = np.zeros((3, 12, 2)); y = cv + .1; p = y[:, None]
    r = solve(cv, p, np.ones((3, 1), bool), np.full(3, -1), y, np.ones((3, 12), bool),
              np.ones(3), 'r', 1., 2., fixed=True)
    assert r['reason'] == 'no_selected_count_support'
