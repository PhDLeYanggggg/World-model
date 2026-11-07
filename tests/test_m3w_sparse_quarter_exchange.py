import numpy as np

from src.world_model.m3w_sparse_quarter_exchange import CAP, apply, first_options, roster


def test_support_changes_only_current_query_ranking():
    choices = np.array([0, -1, -1, 0, -1])
    groups = [np.arange(3), np.arange(3, 5)]
    moves, q = roster(groups, choices, np.ones((5, 1), bool))
    pred = np.array([[.2, 0., 0., 0., 0.], [.1, 0., 0., 0., .8], [.9, 0., 0., 0., 0.]])
    raw = apply(choices, moves, *first_options(pred, q, support=False), 0.)
    aided = apply(choices, moves, *first_options(pred, q, support=True), 0.)
    np.testing.assert_array_equal(raw, [-1, 0, -1, -1, 0])
    np.testing.assert_array_equal(aided, [-1, -1, 0, -1, 0])
    for original, changed in zip(choices, apply(choices, moves, *first_options(pred, q, support=True), None)):
        assert original == changed


def test_cap_is_chronological_and_negative_gain_cannot_switch():
    choices = np.tile([0, -1], 7)
    groups = np.arange(14).reshape(7, 2)
    moves, q = roster(groups, choices, np.ones((14, 1), bool))
    pred = np.zeros((7, 5))
    pred[:, 0] = np.arange(7)-1
    z = apply(choices, moves, *first_options(pred, q, support=True), 0.)
    assert (z[1::2] >= 0).sum() == CAP
    np.testing.assert_array_equal(np.flatnonzero(z[1::2] >= 0), np.arange(2, 2+CAP))
