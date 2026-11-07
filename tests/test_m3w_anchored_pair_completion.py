import numpy as np

from src.evaluation.m3w_anchored_pair_completion import occurrence_terms
from src.evaluation.m3w_shared_pair_completion import shared_bound


def test_one_unknown_endpoint_shares_a_target_across_known_neighbors():
    # The same unknown agent 20 is left on edge 0 and right on edge 1.
    i, j = np.array([20, 30]), np.array([10, 20])
    left = np.array([[[np.nan, np.nan]], [[100., 10.]]])
    right = np.array([[[50., 10.]], [[np.nan, np.nan]]])
    vl, vr = np.array([[False], [True]]), np.array([[True], [False]])
    p = np.array([[[[2., 0.]], [[44., 0.]]], [[[6., 0.]], [[48., 0.]]]])
    groups, coverage = occurrence_terms(i, j, np.array([12]), p, left, right, vl, vr)
    keys, vectors = groups['one_missing']
    assert coverage == (2, 0)
    assert (keys[0] == keys[1]).all()
    np.testing.assert_array_equal(vectors, [[[52., 10.], [56., 10.]], [[56., 10.], [52., 10.]]])
    assert shared_bound(keys, *vectors, np.ones(2)) == (8., 0., 2, 1)
    y = np.array([13., 19.])
    # Recover both original relative-vector errors with one missing point.
    truth = np.stack((y-right[0, 0], left[1, 0]-y))
    np.testing.assert_allclose(np.linalg.norm(p[:, :, 0]-truth, axis=2),
                               np.linalg.norm(vectors-y, axis=2))


def test_two_unknown_endpoints_retain_pair_identity_and_orientation():
    ids_l, ids_r = np.array([20, 30]), np.array([10, 20])
    valid = np.zeros((2, 1), dtype=bool)
    label = np.full((2, 1, 2), np.nan)
    p = np.array([[[[2., 1.]], [[4., 3.]]]])
    groups, coverage = occurrence_terms(ids_l, ids_r, np.array([12]), p, label, label, valid, valid)
    keys, vectors = groups['two_missing']
    assert coverage == (0, 2)
    np.testing.assert_array_equal(keys, [[10, 20, 12], [20, 30, 12]])
    np.testing.assert_array_equal(vectors, [[[-2., -1.], [-4., -3.]]])
    assert len(groups['one_missing'][0]) == 0
