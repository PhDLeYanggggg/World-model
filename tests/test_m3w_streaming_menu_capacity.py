from itertools import product

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye

from src.evaluation import m3w_streaming_menu_capacity as m


def test_token_inequalities_agree_with_exact_replay():
    eligible = np.array([True]*4+[False]+[True]*6)
    rows, upper = m.token_rows(eye(len(eligible), format="csc"), eligible)
    q = len(eligible)
    for count in product(range(3), repeat=3):
        c = np.zeros(q, dtype=int)
        c[[3, 5, 10]] = count
        lp = linprog(np.zeros(q), A_ub=rows[:, q:], b_ub=upper-rows[:, :q]@c, bounds=(0, 10), method="highs")
        assert lp.success == m.respects_tokens(c, eligible)


def test_recording_total_can_edit_early_but_tokens_cannot():
    cv = np.zeros((10, 2, 2))
    ols = cv.copy()
    ols[..., 0] = 2.
    y = cv.copy()
    y[..., 0] = .1
    y[0, :, 0] = .4
    bank = ols[:, None]*.25
    valid = np.ones(y.shape[:2], dtype=bool)
    q = np.repeat(np.arange(5), 2)
    args = cv, ols, bank, y, valid, q, "test", (.5, 1.), np.ones(5, dtype=bool), np.ones(5, dtype=int)
    assert m.assess(*args)["status"] == "verified_positive_both"
    assert m.assess(*args, streaming=True)["status"] == "risk_infeasible"


def test_nested_menu_requires_first_edit():
    cv = np.zeros((15, 2, 2))
    ols = cv.copy()
    ols[..., 0] = 2.
    y = cv.copy()
    y[..., 0] = .1
    y[1, :, 0] = .4
    bank = ols[:, None]*.25
    valid = np.ones(y.shape[:2], dtype=bool)
    q = np.repeat(np.arange(5), 3)
    args = cv, ols, bank, y, valid, q, "test", (.5, 1.), np.ones(5, dtype=bool), np.full(5, 2)
    assert m.assess(*args)["status"] == "verified_positive_both"
    assert m.assess(*args, nested=np.c_[np.arange(0, 15, 3), np.arange(1, 15, 3)])["status"] == "risk_infeasible"
