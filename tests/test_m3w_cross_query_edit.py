import numpy as np
import torch

from scripts.run_m3w_cross_query_edit import schedule


def packet(q, rows, gains, capacity):
    n = len(rows)
    v = np.zeros((n, 6))
    v[:, :2] = gains
    return q, np.array(rows), np.zeros(n, int), v, np.ones(n), 1., capacity


def test_split_and_same_start_share_the_first_edit_but_not_the_second_clock():
    x = [packet(0, [0, 1], [[5., 4.], [2., 2.]], 2), packet(1, [2], [[1., 1.]], 1)]
    y = schedule(x)
    assert y["same"] == [(0, 0, 0), (0, 1, 0)]
    assert y["split_same_start"] == [(0, 0, 0), (1, 2, 0)]
    assert schedule(x[:1])["split_same_start"] == [(0, 0, 0)]


def test_joint_carries_gain_deficit_with_identical_clock_and_first_choice():
    x = [packet(0, [0], [[9., 0.]], 1), packet(2, [1, 2], [[1., 8.], [5., 5.]], 2)]
    y = schedule(x)
    assert y["early_independent"] == [(0, 0, 0), (2, 2, 0)]
    assert y["early_joint"] == [(0, 0, 0), (2, 1, 0)]
    assert schedule(x[:1])["early_joint"] == y["early_joint"][:1]


def test_two_arms_of_one_row_cannot_trigger_two_edit_pair():
    x = [packet(0, [0, 0], [[9., 0.], [2., 2.]], 2)]
    y = schedule(x)
    assert not y["same"] and not y["split_same_start"]
    assert y["early_independent"] == y["early_joint"] == [(0, 0, 0)]


def test_cross_query_scores_share_the_scale_fixed_by_first_query():
    a = packet(0, [0], [[9., 0.]], 1)
    b = packet(1, [1, 2], [[.015, .015], [0., 0.]], 2)
    b[3][:, 5] = [-2.2, 2.2]
    shifted = (*b[:5], .1, b[6])
    before, after = schedule([a, b]), schedule([a, shifted])
    assert all(before[k] == after[k] for k in ("early_independent", "early_joint"))
    assert after["early_independent"][1] == (1, 2, 0)
