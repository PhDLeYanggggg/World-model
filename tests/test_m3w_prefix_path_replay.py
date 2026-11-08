import numpy as np

from src.world_model.m3w_prefix_path_replay import choose, prefix_input, rank_counts


def test_prefix_cannot_see_its_own_replay_outcomes():
    h = np.arange(16).reshape(1, 8, 2).astype(float)
    changed = h.copy()
    changed[:, 5:] += 999
    for mode in ("linear", "stationary"):
        prefix = prefix_input(h, 5, mode)
        np.testing.assert_array_equal(prefix, prefix_input(changed, 5, mode))
        np.testing.assert_array_equal(prefix[:, -5:], h[:, :5])


def test_fallback_and_concordance_do_not_count_unavailable_or_tied_rows():
    a = np.array([[0., 0., 2.], [3., 1., 0.], [2., 1., 0.]])
    b = np.array([[0., 0., 1.], [0., 0., 1.], [2., 1., 0.]])
    p = choose({"linear": (a, np.array([4, 4, 0])), "stationary": (b, np.full(3, 4))}, 1)
    assert p["linear_prefix"].tolist() == [2, 0, 1]
    assert p["padding_consensus"].tolist() == [2, 1, 1]
    assert rank_counts(a, b, np.array([True, True, False])) == {"comparable": 4, "concordant": 2, "rows": 2}
