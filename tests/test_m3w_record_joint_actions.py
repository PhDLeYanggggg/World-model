import numpy as np

from src.world_model import m3w_record_joint_actions as m


def sample():
    rng = np.random.default_rng(42)
    history = rng.normal(size=(8, 8, 2)).cumsum(1)
    floor = np.arange(1, 13)[None, :, None] * np.diff(history, axis=1)[:, -1:]
    paths = floor[:, None] + rng.normal(size=(8, 3, 12, 2))
    return history, paths, floor


def test_costs_are_finite_with_unknown_and_exact_floor_action():
    h, paths, floor = sample()
    paths[:, 0] = floor
    x, scale = m.descriptors(h, paths, floor, np.full(8, 8))
    valid = np.ones((8, 12), dtype=bool); valid[:3] = False
    target = paths[:, 1].copy(); target[:3] = np.nan
    c = m.coefficients([m.fitting_moments(x, paths, floor, target, valid, scale, np.ones(8), 1.)])
    utility, a, e, legal = m.decision_costs(x, c, paths, floor, scale)
    assert np.isfinite([utility, a, e]).all()
    assert not legal[:, 0].any()
    np.testing.assert_array_equal(utility[:, 0], 0.)


def test_counts_and_joint_objective_match_exhaustive_pair_search():
    h, paths, floor = sample()
    u = np.arange(24).reshape(8, 3) / 10
    legal = np.ones((8, 3), dtype=bool)
    scale = np.ones(8); ols = floor * .8
    choices = m.choose_query(h, floor, paths, ols, scale, u, legal)
    np.testing.assert_array_equal((choices >= 0).sum(1), 2)
    assert len(set(choices[1, choices[1] >= 0])) == 1
    i, j = m.local_edges(h)
    deg = np.bincount(np.r_[i, j], minlength=len(h))
    w = 1. / np.sqrt(deg[i] * deg[j])
    def score(c):
        p = m.selected_paths(c, paths, floor)
        difference = p[i] - p[j] - (ols[i] - ols[j])
        sel = np.flatnonzero(c >= 0)
        return u[sel, c[sel]].sum() - m.PAIR_WEIGHT * (w * (difference**2).sum(2).mean(1)).sum()
    trials = []
    for a in range(8):
        for b in range(a+1, 8):
            t = np.full(8, -1); t[[a, b]] = 2
            trials.append(score(t))
    np.testing.assert_allclose(score(choices[3]), max(trials))


def test_budget_prefix_uses_both_predicted_constraints_and_can_abstain():
    choice = np.zeros(4, dtype=np.int16)
    u = np.array([[4.], [3.], [2.], [1.]])
    a = np.array([[1.], [-2.], [.5], [.5]])
    e = np.array([[1.], [-2.], [1.5], [.5]])
    np.testing.assert_array_equal(m.budget_prefix(choice, u, a, e, np.ones(4)), [0, 0, -1, -1])
    np.testing.assert_array_equal(m.budget_prefix(choice, u, a, np.ones_like(e), np.ones(4)), -1)


def test_empty_pair_support_remains_unknown():
    h, _, floor = sample()
    assert m.pair_error(h, floor, floor, np.zeros((len(h), 12), bool)) == (None, 0)
