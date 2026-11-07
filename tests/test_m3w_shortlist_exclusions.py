import numpy as np

from src.world_model.m3w_shortlist_exclusions import alternatives, mechanism, mix


def test_mix_keeps_exact_six_actions_then_adds_unique_actions():
    q = np.arange(10)
    x, y = np.ones((8, 12)), np.full((8, 12), 2.)
    a = q[:8]
    b = np.array([0, 8, 1, 9, 2, 3, 4, 5])
    k = np.zeros(8, int)
    _, i, _, z = mix((q, a, k, x), (q, b, k, y))
    np.testing.assert_array_equal(i, [0, 1, 2, 3, 4, 5, 8, 9])
    np.testing.assert_array_equal(z[:, 9], np.arange(8)/8)
    np.testing.assert_array_equal(z[:, 1], [1, 1, 1, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(x[:, 9], 1.)


def test_drop_agent_preference_does_not_drop_unique_action_constraint():
    s = np.arange(24, dtype=float).reshape(12, 2)
    legal = np.ones_like(s, bool)
    row, cost = alternatives(s, s, -s, -s, legal)
    assert row[0] == cost[0] == 23
    assert len(np.unique(row//2)) == 8
    assert len(np.unique(cost)) == 8
    assert len(np.unique(cost//2)) < 8
    assert mechanism(True, False) == 'cost_alone_sufficient'
