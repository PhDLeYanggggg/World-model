from itertools import product

import numpy as np
import pytest

from src.evaluation.m3w_count_motion_capacity import allocate, movement_prefix
from src.evaluation.m3w_streaming_menu_capacity import respects_tokens


@pytest.mark.parametrize('total,cap', [(1, 3.), (2, 5.)])
def test_movement_capped_objectives_match_exhaustive_search(total, cap):
    legal = np.tile([True, True, False], (11, 1))
    legal[-2:, 2] = True
    q = np.arange(len(legal))
    moves = np.column_stack((np.zeros(len(q)), 1. + q % 4, 4. + q % 4))
    gain = np.zeros((*legal.shape, 2))
    gain[:, 1] = np.column_stack((q, 12 - q))
    gain[:, 2] = [20., 20.]
    risk = np.zeros((*legal.shape, 3))
    risk[7, 1, 2] = .04
    easy = legal.copy()
    easy[:, 0] = False
    offset = np.array([0., -.5])
    scores = np.zeros_like(moves)
    scores[:, 1] = q
    scores[:, 2] = 100.
    best_score, best_gain = -np.inf, -np.inf
    for counts in product(*(np.flatnonzero(row) for row in legal)):
        if sum(counts) != total or moves[q, counts].sum() > cap or not respects_tokens(counts, legal[:, 1]):
            continue
        best_score = max(best_score, scores[q, counts].sum())
        if (risk[q, counts].sum(0) <= [0., 0., .02]).all() and easy[q, counts].any():
            best_gain = max(best_gain, (offset + gain[q, counts].sum(0)).min())

    score_plan, status = allocate(legal, moves, total, cap, score=scores)
    assert status == 'score_optimal'
    assert scores[q, score_plan].sum() == pytest.approx(best_score)
    witness, status = allocate(legal, moves, total, cap, terms=(gain, risk, easy, offset, .02, 100.))
    assert status == 'candidate'
    assert (offset + gain[q, witness].sum(0)).min() == pytest.approx(best_gain)
    assert moves[q, witness].sum() <= cap and witness.sum() == total


def test_peak_budget_counts_agents_not_steps():
    cv = np.zeros((3, 2, 2))
    paths = np.array([[[3., 4.], [0., 1.]], [[1., 0.], [0., 3.]], [[2., 0.], [2., 0.]]])
    np.testing.assert_array_equal(movement_prefix(cv, paths, np.array([[0, 1], [2, -1]])), [[0, 5, 8], [0, 2, 2]])
    legal = np.tile([True, True, False], (5, 1))
    movement = np.tile([0., 1., 0.], (5, 1))
    _, status = allocate(legal, movement, 0, 0., score=np.zeros_like(movement))
    assert status == 'zero_realized'


def test_exact_boundary_survives_very_close_unaffordable_options():
    relative = [1.9568516173073789, 1.3775198746852153, .687874061044493,
        .4685212771163076, 1., .41905817453069333, 1.0000000109600806,
        1.000000047474729, 1.391612515419229, 1.69935426230854, .5319086593254186,
        2.9648345628907764, 1.0312199576890488]
    legal = np.tile([True, True, False], (len(relative), 1))
    movement = np.column_stack((np.zeros(len(relative)), relative, relative))
    scores = np.zeros_like(movement)
    scores[4:8, 1] = [.81, -1.18, 2.34, 1.02]
    result, status = allocate(legal, movement, 1, 1., score=scores)
    assert status == 'score_optimal' and np.flatnonzero(result).tolist() == [4]
