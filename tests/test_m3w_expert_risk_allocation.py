import numpy as np

from src.world_model.m3w_expert_risk_allocation import choose, costs, selected_paths, targets


def test_missing_steps_debit_known_zero_change():
    floor = np.zeros((2, 12, 2))
    delta = np.zeros((2, 3, 12, 2))
    delta[:, 1, :, 0] = 2
    missing = np.full_like(floor, np.nan)
    value = targets(floor, delta, missing, np.zeros((2, 12), bool), np.ones(2), np.array([1, 0]))
    np.testing.assert_allclose(value[:, 0], 0.)
    np.testing.assert_allclose(value[:, 1, 0], -2. / 12.)
    np.testing.assert_allclose(value[:, 1, 1], 2. / 12.)
    np.testing.assert_allclose(value[:, 1, 2], [2. / 12., 0.])


def test_pair_cost_cancels_when_both_move_together():
    base = np.array([[[0., 0.]], [[10., 0.]]])
    single, both = costs(base, base + [[[3., 0.]], [[3., 0.]]])
    assert single[0, 1] == 3.
    assert both[0, 1] == 0.


def test_matched_legal_counts_and_identity():
    ranks = np.array([[2., 1.], [1., 2.], [3., 4.], [0., 0.], [2., 0.]])
    legal = np.array([[1, 0], [1, 1], [1, 1], [0, 0], [1, 1]], bool)
    q = [np.arange(4), np.array([4])]
    base = np.zeros((5, 12, 2))
    delta = np.ones((5, 2, 12, 2))
    selections = choose(ranks, legal, q, base, delta, np.column_stack((np.arange(5), np.zeros(5))))
    for selection in selections.values():
        assert [(selection[i] >= 0).sum() for i in q] == [2, 1]
        assert selection[3] == -1
        assert legal[np.flatnonzero(selection >= 0), selection[selection >= 0]].all()
        np.testing.assert_array_equal(selected_paths(base, delta, selection)[3], base[3])
