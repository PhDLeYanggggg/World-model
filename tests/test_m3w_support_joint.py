import numpy as np

from src.world_model.m3w_support_joint import retrieve, select


def test_retrieve_exact_budget_and_no_illegal_actions():
    r = np.random.default_rng(12)
    legal = r.random((9, 3)) > .3
    scores = [r.normal(size=legal.shape) for _ in range(4)]
    result = retrieve(legal, *scores)
    assert len(set(result)) == 8 and legal.ravel()[result].all()


def test_insert_and_delete_same_query_when_full_no_future_funding():
    direct = np.array([0, 0, -1, 0, -1])
    q = np.arange(3)
    a = np.array([[2, 0]])
    groups = [(q, a, a), (np.array([3, 4]), np.empty((0, 2), int), np.empty((0, 2), int))]
    est = tuple(np.full((5, 1), v) for v in (1., -.1, -.1, .2, .7))
    rem = np.array([[1., .2, .2, 0.], [0., 0., 0., .1],
                    [0., 0., 0., 0.], [1., .3, .3, 0.], [0., 0., 0., 0.]])
    np.testing.assert_array_equal(select(direct, groups, est, rem, 'support_insert', (1., 0.)), direct)
    z = select(direct, groups, est, rem, 'support_joint', (1., 0.))
    np.testing.assert_array_equal(z, [-1, 0, 0, -1, -1])
    np.testing.assert_array_equal(select(direct, groups[:1], est, rem, 'support_joint', (1., 0.))[:3], z[:3])
