import numpy as np

from src.world_model import m3w_cv_joint_selector as model


def test_joint_changes_only_recipients_at_exact_counts():
    score = np.array([[3., 2.], [2., 1.], [1., 0.]])
    legal = np.ones_like(score, bool)
    edges = np.array([[0, 0], [1, 2]])
    result = model.choose(score, legal, edges, np.zeros(2), np.zeros(2), np.array([-10., 10.]), 2, 1.)
    np.testing.assert_array_equal(result['joint'], [0, -1, 0])
    np.testing.assert_array_equal(result['independent'], [0, 0, -1])
    assert all((r >= 0).sum() == 2 for r in result.values())


def test_pair_features_and_values_are_exchange_invariant():
    rng = np.random.default_rng(20)
    a, b = rng.normal(size=(2, 5, 12, 2))
    c, d = rng.normal(size=(2, 5, 12, 2))
    h, g = rng.normal(size=(2, 5, 8, 2))
    mask = rng.uniform(size=(5, 12)) > .4
    x, z = model.pair_context(h, g, c, d), model.pair_context(g, h, d, c)
    np.testing.assert_allclose(model.pair_features(x, a, b), model.pair_features(z, b, a), atol=1e-6)
    np.testing.assert_allclose(model.pair_target(c-d, a-b, mask, a, b, x[-1]),
                               model.pair_target(d-c, b-a, mask, b, a, z[-1]))


def test_missing_pair_steps_pay_distance_not_zero():
    p = np.zeros((2, 12, 2))
    a = np.ones_like(p)
    v = np.zeros((2, 12), dtype=bool)
    np.testing.assert_allclose(model.pair_target(p, p, v, a, p, np.ones(2)), -np.sqrt(2.))


def test_radius_matching_preserves_counts_and_equalizes_each_query():
    cv = np.zeros((4, 12, 2))
    a, b = cv.copy(), cv.copy()
    a[[0, 2]] = 5.
    b[[1, 3]] = 3.
    m = model.matched_radius(cv, dict(a=a, b=b), np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(m['a'] != 0., a != 0.)
    np.testing.assert_allclose(m['a'][[0, 2]], m['b'][[1, 3]])
