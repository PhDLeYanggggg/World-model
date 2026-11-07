import numpy as np
import pytest

from src.evaluation import m3w_joint_slate_capacity as m


def fixture():
    cv = np.zeros((4, 12, 2))
    y = cv.copy()
    y[:, :, 0] = np.arange(1., 5.)[:, None]
    valid = np.ones((4, 12), bool)
    offsets = np.array([0, 2, 4])
    slots = np.full((4, 2, 2), -1, dtype=int)
    paths = np.zeros((*slots.shape, 12, 2))
    return cv, y, valid, offsets, slots, paths


def test_adding_a_safe_easy_option_restores_support_but_not_necessarily_ols4_gain():
    cv, y, valid, offsets, slots, paths = fixture()
    slots[3, 1, 0] = 2
    paths[3, 1, 0] = y[2]
    slots[0, 0, 0] = 0
    paths[0, 0, 0] = y[0]
    out = m.audit(cv, y.copy(), y, valid, offsets, slots, paths, 'fixture', 1.5, 3.)['outcomes']
    assert out['joint_only_vs_cv']['status'] == 'no_easy_option_support'
    assert out['four_option_vs_cv']['verified_witness'] and out['four_option_vs_cv']['witness_positive']
    assert out['four_option_vs_ols4']['verified_witness'] and not out['four_option_vs_ols4']['witness_positive']


def test_unknown_future_radius_is_charged_to_both_budgets():
    cv, y, valid, offsets, slots, paths = fixture()
    valid[1] = False
    slots[3, 0] = [0, 1]
    paths[3, 0] = y[:2]
    result = m.audit(cv, cv, y, valid, offsets, slots, paths, 'fixture', 1.5, 3.)
    assert result['outcomes']['joint_only_vs_cv']['proved_infeasible']


def test_reject_options_from_another_query():
    cv, y, valid, offsets, slots, paths = fixture()
    slots[0, 0, 0] = 2
    with pytest.raises(ValueError, match='another query'):
        m.prepare(cv, cv, y, valid, offsets, slots, paths, 1.5)
