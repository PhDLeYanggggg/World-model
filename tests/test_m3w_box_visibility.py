import numpy as np
import pytest

from src.world_model.m3w_box_visibility import aligned, frame_features


def test_overlap_is_normalized_by_target_box_and_ignores_self():
    boxes = np.array([[0., 0., 4., 4.], [1., 1., 2., 2.]])
    x = frame_features(boxes)
    np.testing.assert_allclose(x[:, 3], [.25, 1.])
    assert frame_features(boxes[:1])[0, 3] == 0.


def test_future_annotation_changes_do_not_change_earlier_query_features():
    rows = np.array([[f, a, a, 0., 2., 2.] for f in range(9) for a in (1, 2)], float)
    xy = np.broadcast_to([2., 1.], (1, 8, 2))
    x = aligned(rows, np.array([1]), np.array([7]), xy)
    rows[rows[:, 0] == 8, 2:6] = [99., 99., 8., 8.]
    np.testing.assert_array_equal(aligned(rows, np.array([1]), np.array([7]), xy), x)
    assert x.shape == (1, 27)


def test_box_identity_and_center_must_match_frozen_trajectory():
    rows = np.array([[f, 1, 0., 0., 2., 2.] for f in range(8)])
    with pytest.raises(AssertionError):
        aligned(rows, np.array([1]), np.array([7]), np.zeros((1, 8, 2)))
    with pytest.raises(ValueError, match='identity'):
        aligned(rows, np.array([2]), np.array([7]), np.ones((1, 8, 2)))
