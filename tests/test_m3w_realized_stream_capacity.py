from itertools import product

import numpy as np
import pytest

from src.evaluation.m3w_realized_stream_capacity import solve
from src.evaluation.m3w_streaming_menu_capacity import respects_tokens


def toy():
    legal = np.tile([True, True, False], (10, 1))
    legal[-1, 2] = True
    gain = np.zeros((10, 3, 2))
    gain[:, 1] = np.c_[np.arange(10) + 1., np.arange(10) + .5]
    gain[0, 1] = [100., 100.]
    gain[-1, 2] = [30., 30.]
    risk = np.zeros((10, 3, 3))
    risk[-1, 2, 1] = .04
    easy = legal.copy()
    easy[:, 0] = False
    return gain, risk, easy, legal, np.array([0., -1.]), .02, 100.


@pytest.mark.parametrize('total', [1, 2])
def test_exact_budget_matches_exhaustive_prefix_choice(total):
    g, r, e, legal, base, ec, mass = args = toy()
    opt = -np.inf
    rows = np.arange(len(legal))
    for choice in product(*(np.flatnonzero(row) for row in legal)):
        if sum(choice) != total or not respects_tokens(choice, legal[:, 1]):
            continue
        if (r[rows, choice].sum(0) > [0., 0., ec]).any() or not e[rows, choice].any():
            continue
        opt = max(opt, float(100.*(base+g[rows, choice].sum(0)).min()/mass))
    count, info = solve(*args, total)
    assert info['status'] == 'candidate'
    assert count.sum() == total and count[0] == 0
    assert info['witness_pp'] == pytest.approx(opt)
    assert info['upper_pp'] == pytest.approx(opt)


def test_zero_realized_count_is_not_a_witness():
    count, info = solve(*toy(), 0)
    assert count is None and info['status'] == 'zero_realized'
