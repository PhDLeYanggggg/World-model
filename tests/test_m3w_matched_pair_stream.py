import numpy as np

from src.evaluation.m3w_matched_pair_stream import blocks, update_best


def test_stream_order_matches_original_and_bounds_batch_size():
    a = np.array([0, 0, 1, 2, 2, 3])
    i, j = np.triu_indices(len(a), 1)
    keep = a[i] != a[j]
    v = list(blocks(a, 4))
    assert all(0 < len(left) <= 4 for left, _ in v)
    np.testing.assert_array_equal(np.concatenate([left for left, _ in v]), i[keep])
    np.testing.assert_array_equal(np.concatenate([right for _, right in v]), j[keep])


def test_first_tie_and_no_finite_same_family():
    best = {}
    update_best(best, np.array([0, 0]), np.array([1, 2]), {'same': np.full(2, -np.inf), 'joint': np.array([4., 4.])})
    update_best(best, np.array([1]), np.array([2]), {'same': np.array([1.]), 'joint': np.array([4.])})
    assert best['joint'][0].tolist() == [0, 1]
    assert best['same'][0].tolist() == [1, 2]
