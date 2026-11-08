import numpy as np

from scripts.run_m3w_acme_camera_repair import associate_labels, floor_filter


def test_floor_filter_uses_ground_anchor_not_box_center():
    floor = np.array([[0, 100], [100, 100], [100, 140], [0, 140]], dtype=np.float32)
    boxes = np.array([[10, 40, 30, 120, .8, 1], [40, 20, 60, 80, .99, 1]])
    np.testing.assert_array_equal(floor_filter(boxes, [floor]), boxes[:1])


def test_matching_is_one_to_one_and_requires_local_support():
    tracks = np.array([[4, 0, 0, 20, 20, .7, 1], [5, 200, 200, 220, 220, .7, 1]])
    labels = [("p", [10, 20]), ("q", [11, 20]), ("r", [70, 60])]
    assert associate_labels(tracks, labels) == [(4, "p")]
