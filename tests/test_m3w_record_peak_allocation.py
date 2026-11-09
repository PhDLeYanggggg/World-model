import itertools

import numpy as np
import pytest

from src.evaluation import m3w_record_peak_allocation as m


def test_segment_envelope_matches_brute_force_on_integer_caps():
    caps = np.array([[1., 4., 3.], [2., 5., 1.], [0., 1., 2.]])
    w = np.array([.5, .5, 1.])
    best = max((np.minimum(caps, np.array(r)[:, None]).sum(1) @ w
                for r in itertools.product(range(5), repeat=3) if sum(r) <= 4))
    assert m.radius_envelope(caps, w, 4.) == pytest.approx(best)
    assert m.radius_envelope(caps, w, 0.) == 0.


def test_missing_ols_label_credits_only_motion_toward_reference():
    cv = np.zeros((2, 2, 2))
    ols = cv.copy()
    ols[:, 1, 0] = 4.
    y = cv.copy()
    y[:, 0, 0] = 1.
    valid = np.array([[True, False], [True, False]])
    out = m.arbitrary_bound(cv, ols, y, valid, np.array([0, 1]), np.array([0., 2.]))
    assert out['query'] == pytest.approx([50., -250.])
    assert out['record'] == pytest.approx([100., -200.])


def test_pooled_budget_can_move_sole_edit_but_fixed_query_cannot():
    cv = np.zeros((3, 2, 2))
    y = cv.copy()
    y[..., 0] = 2.
    valid = np.ones((3, 2), dtype=bool)
    ols = cv.copy()
    ols[1, :, 0] = .25
    bank = cv[:, None].copy()
    bank[:, 0, :, 0] = [-1., -1.]  # overwritten on the helpful query below
    bank[1, 0, :, 0] = 1.
    q = np.array([0, 1, 2])
    budgets = (np.array([1, 0, 0]), np.array([1., 0., 0.]))
    for mode in ('query', 'record'):
        d = m.design(cv, ols, bank, y, valid, q, 3., *budgets, mode)
        take, status = m.solve(d, seconds=2.)
        if mode == 'query':
            assert take is None
            assert status['status'] in ('risk_infeasible', 'nonpositive_bound')
        else:
            assert d['row'][take].tolist() == [1]
            assert status['upper_pp'] > 0.


def test_dominance_does_not_drop_cheaper_path():
    cv = np.zeros((1, 2, 2))
    bank = np.repeat(cv[:, None], 2, axis=1)
    bank[:, 0, :, 0] = 1.
    bank[:, 1, :, 0] = 2.
    y = cv.copy()
    y[..., 0] = 3.
    d = m.design(cv, cv, bank, y, np.ones((1, 2), dtype=bool), np.zeros(1),
                 4., np.array([1]), np.array([3.]), 'record')
    assert d['arm'].tolist() == [0, 1]
