import numpy as np

from src.world_model.m3w_moving_sparse_support import targets


def test_known_easy_positive_gain_is_necessary_and_strong_offset_matters():
    t = targets([2., 2., 3., 2.], np.array([[1., 1.9], [1., 2.], [1., 2.], [1., 2.]]),
                [True, False, True, True], 2., np.array([.5, 1., 1., 1.]), -.6)
    np.testing.assert_array_equal(t[..., 0], [[True, True], [False, False], [False, False], [True, False]])
    np.testing.assert_array_equal(t[..., 1], [[False, False], [False, False], [False, False], [True, False]])
