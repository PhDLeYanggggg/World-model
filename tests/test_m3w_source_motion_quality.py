import numpy as np
import pytest

from src.evaluation.m3w_source_motion_quality import (
    BOX_EDGES, PIXEL_EDGES, bins, direction_scores, neighbor_directions,
    past_box_features, trajectory_quality,
)


def test_box_features_translation_and_scale_invariant():
    boxes = np.tile([0., 0., 4., 6.], (2, 8, 1))
    boxes[1, :, 2] += np.arange(8)*.2
    a = past_box_features(boxes)
    assert a.shape == (2, 38)
    np.testing.assert_allclose(a, past_box_features(boxes*7+121), atol=1e-6)
    assert not np.array_equal(a[0], a[1])
    with pytest.raises(ValueError): past_box_features(boxes[:, :7])
    with pytest.raises(ValueError): past_box_features(np.zeros_like(boxes))


def test_magnitude_labels_do_not_equate_nonzero_with_half_box_motion():
    y = np.zeros((4, 12, 2))
    y[1, 0, 0] = .5
    y[2, :, 0] = np.arange(1, 13)
    y[3, :, 1] = 12
    q = trajectory_quality(y, np.full(4, 20.))
    np.testing.assert_array_equal(q['nonzero'], [False, True, True, True])
    np.testing.assert_array_equal(q['half_box_excursion'], [False, False, True, True])
    np.testing.assert_array_equal(q['returned_to_origin'], [False, True, False, False])
    np.testing.assert_array_equal(q['final_four_outside_half_box'], [False, False, False, True])
    assert q['first_changed_step'].tolist() == [0, 1, 1, 1]


@pytest.mark.parametrize('edges', [PIXEL_EDGES, BOX_EDGES])
def test_bins_retain_boundary_and_all_rows(edges):
    values = np.r_[edges, np.asarray(edges)+.00001]
    b = bins(values, edges)
    np.testing.assert_array_equal(np.stack(list(b.values())).sum(0), 1)
    assert b['zero'].sum() == 1
    with pytest.raises(ValueError): bins(np.array([-1.]), edges)


def test_neighbor_direction_uses_only_valid_current_past_pairs():
    g = np.zeros((1, 476))
    p = g[:, 38:166].reshape(1, 8, 8, 2)
    t = g[:, 166:230].reshape(1, 8, 8)
    m = g[:, 230:294].reshape(1, 8, 8)
    p[0, 1, -2:] = [[1, 0], [2, 0]]
    t[0, 1, -2:] = [-1, 0]; m[0, 1, -2:] = 1
    p[0, 0] = 100
    hints = neighbor_directions(g)
    np.testing.assert_array_equal(hints['nearest_moving_neighbor'], [[1, 0]])
    support, cosine = direction_scores(hints['nearest_moving_neighbor'], [[0, 1]])
    assert support[0] and cosine[0] == 0
    assert not direction_scores(np.zeros((1, 2)), [[1, 1]])[0][0]
