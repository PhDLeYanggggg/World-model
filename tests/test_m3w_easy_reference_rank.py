import numpy as np

from src.world_model.m3w_easy_reference_rank import motion, retrieve


def test_history_uses_no_later_group_and_ignores_translation():
    rng = np.random.default_rng(3)
    h = rng.normal(size=(4, 8, 2))
    g = [np.arange(2), np.arange(2, 4)]
    a = motion(h, g)
    np.testing.assert_allclose(a, motion(h+10., g))
    h[2:] *= 9
    np.testing.assert_array_equal(a[:2], motion(h, g)[:2])


def test_retrieval_prefers_distinct_agents_and_maintains_legal_budget():
    q = np.arange(4)
    legal = np.ones((4, 3), bool)
    legal[0, 1:] = False
    gain = np.arange(12).reshape(4, 3)
    rows = retrieve([q], legal, np.arange(4), -gain, gain)[0]
    assert len(rows) == 8
    assert len(np.unique(rows[:4, 0])) == 4
    assert legal[tuple(rows.T)].all()
