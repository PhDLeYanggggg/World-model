import numpy as np
import pytest

from src.evaluation.m3w_constant_feature_support import clamp_constant_support


def test_constant_repair_preserves_every_training_row_and_variable_dimensions():
    train = np.array([[120, 10, .3], [120, 10, .4]], np.float32)
    same, constant = clamp_constant_support(train, train)
    assert np.array_equal(same, train)
    assert constant.tolist() == [True, True, False]
    query = np.array([[72, 6, .6]], np.float32)
    changed, _ = clamp_constant_support(train, query)
    np.testing.assert_array_equal(changed, [[120, 10, np.float32(.6)]])
    np.testing.assert_array_equal(query, [[72, 6, np.float32(.6)]])


def test_nearly_constant_feature_is_not_silently_thresholded():
    train = np.array([[1., 2.], [1. + 1e-12, 2.]])
    _, constant = clamp_constant_support(train, np.array([[0., 0.]]))
    assert constant.tolist() == [False, True]
    with pytest.raises(ValueError):
        clamp_constant_support(train, np.array([[np.nan, 0.]]))
