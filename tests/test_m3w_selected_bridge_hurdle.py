import numpy as np

from src.world_model.m3w_selected_bridge_hurdle import bank_decode, matched_independent, targets


def test_support_hurdle_changes_action_without_matching_future():
    p = np.array([[[4, 4, .8, .05]], [[3, 3, .8, .05]], [[1, 1, .8, .8]]])
    legal, q = np.ones((3, 1), bool), np.zeros(3, int)
    ungated, gated = bank_decode(p, legal, q, [(.5, 0.), (.5, .5)])
    assert ungated.tolist() == [0, 0, -1]
    assert gated.tolist() == [0, -1, 0]
    np.testing.assert_array_equal(matched_independent(p, legal, q, gated), ungated)


def test_unknown_is_not_an_admissible_easy_anchor_even_if_excess_zero():
    m = np.zeros((3, 1, 6))
    m[1:, 0, 4] = 1.
    m[2, 0, 5] = 1.
    assert targets(m)[:, 0, 2:].tolist() == [[0., 0.], [1., 0.], [1., 1.]]
