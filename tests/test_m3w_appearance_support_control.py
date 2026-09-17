import numpy as np
import pytest

from src.evaluation.m3w_appearance_support_control import fit_box, project_box


def test_training_rows_unchanged_and_constant_features_retained():
    x = np.array([[1., 2., 5.], [3., 4., 5.]])
    lower, upper = fit_box(x)
    result, outside = project_box(x, lower, upper, np.arange(3))
    np.testing.assert_array_equal(result, x)
    assert not outside.any()
    result, outside = project_box(np.array([[2., 3., 10.]]), lower, upper, [2])
    np.testing.assert_array_equal(result, [[2., 3., 5.]])
    assert outside.tolist() == [True]


def test_selected_columns_only_and_input_not_mutated():
    x = np.array([[-2., 8., 15.], [.5, .5, .5]])
    old = x.copy()
    result, outside = project_box(x, np.zeros(3), np.ones(3), [2])
    np.testing.assert_array_equal(x, old)
    np.testing.assert_array_equal(result[:, :2], x[:, :2])
    assert result[0, 2] == 1 and outside.tolist() == [True, False]


def test_queries_do_not_fit_support_and_batching_does_not_change_it():
    lower, upper = fit_box(np.array([[0., 0.], [1., 1.]]))
    a, m = project_box(np.array([[20., .5]]), lower, upper, [0, 1])
    b, _ = project_box(np.array([[20., .5], [1e10, -1e10]]), lower, upper, [0, 1])
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(upper, [1., 1.])
    assert m[0]


@pytest.mark.parametrize('x', [np.zeros((0, 3)), np.ones(3), np.array([[np.nan]])])
def test_bad_training_features_rejected(x):
    with pytest.raises(ValueError):
        fit_box(x)


def test_bad_projection_rejected():
    for columns in ([3], [-1], [1, 1], [.5]):
        with pytest.raises(ValueError):
            project_box(np.ones((2, 3)), np.zeros(3), np.ones(3), columns)
    with pytest.raises(ValueError):
        project_box(np.ones((2, 3)), np.ones(3), np.zeros(3), [0])
