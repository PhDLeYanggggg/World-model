import numpy as np

from src.world_model.m3w_shortlist_portfolio import retrieve, winners


def test_fixed_budget_and_distinct_agents_with_stable_anchor():
    rng = np.random.default_rng(23)
    j, u, a, e = rng.normal(size=(4, 12, 3))
    valid = rng.uniform(size=j.shape) > .2
    result = retrieve(j, u, a, e, valid)
    assert result[0] == np.where(valid, j, -np.inf).argmax()
    assert len(np.unique(result//3)) == len(result) == 8
    assert valid.ravel()[result].all()
    np.testing.assert_array_equal(result, retrieve(j, u, a, e, valid))


def test_fill_when_few_agents_and_many_illegal_actions():
    a = np.zeros((2, 4))
    legal = np.array([[True, False, True, True], [False, False, False, True]])
    order = retrieve(a, a, a, a, legal)
    assert order[:2].tolist() == [0, 7]
    assert set(order) == {0, 2, 3, 7}
    assert not len(retrieve(a, a, a, a, ~np.ones_like(legal)))


def test_event_and_risk_winners_are_distinct():
    q, i, j = np.arange(2), np.arange(2), np.zeros(2, int)
    x = np.eye(2)
    coef = np.array([[3., 0., 0., 0., .1], [2., 4., 0., 0., .9]])
    assert winners((q, i, j, x), coef) == [(.1, 0, 0), (.9, 1, 0)]
