import numpy as np

from src.world_model import m3w_native_support_retrieval as model


def test_reranking_uses_same_candidates_and_preserves_the_other_action():
    original = np.array([1, 0, -1, -1, -1])
    u = np.array([[0., 6.], [1., 2.], [4., 3.], [3., 5.], [9., 9.]])
    valid = np.ones_like(u, dtype=bool)
    valid[-1] = False
    gain, support = model.rerank(original, u, np.ones(5), valid, np.array([0., .3, .9, .6, 1.]))
    np.testing.assert_array_equal(gain, [1, -1, -1, 1, -1])
    np.testing.assert_array_equal(support, [1, -1, 0, -1, -1])
    np.testing.assert_array_equal(model.rerank(original, u, np.ones(5), valid, np.zeros(5))[1], gain)


def test_held_labels_cannot_affect_support_model():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(7, 4))
    y = np.zeros(7)
    sources = np.array(list('abcbaba'))
    p = model.train(x, y, sources, np.ones(7), 'a')
    y[sources == 'a'] = 1.
    q = model.train(x, y, sources, np.ones(7), 'a')
    for a, b in zip(p, q):
        np.testing.assert_array_equal(a, b)


def test_neighbor_features_stay_within_query():
    h = np.repeat(np.array([[[0., 0.]], [[3., 4.]], [[.1, 0.]]]), 8, axis=1)
    b = np.zeros((3, 2, 12, 2))
    x = model.features(h, b, b[:, 0], np.array([2, 2, 1]), [[0, 1], [2]])
    np.testing.assert_allclose(x[:, 14], np.log1p([5., 5., 0.]))
