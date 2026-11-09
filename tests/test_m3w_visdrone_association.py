import cv2
import numpy as np
import pytest

from scripts.audit_m3w_visdrone_association import possible_reference_support
from src.world_model.m3w_camera_association import MotionAssociation, camera_step
from src.world_model.m3w_streaming_boxes import StreamingTracks


def boxes(*centers):
    return np.array([[x-5, 10, x+5, 20, 1., 1.] for x in centers]).reshape(-1, 6)


def test_velocity_resolves_a_crossing_using_only_prior_detections():
    naive, motion = StreamingTracks(), MotionAssociation(True)
    prior = [boxes(10, 30), boxes(17, 23)]
    for i, d in enumerate(prior):
        original, _ = naive.step(i, d)
        np.testing.assert_array_equal(motion.step(i, d), original)
    current = boxes(24, 16)
    original, _ = naive.step(2, current)
    tracked = motion.step(2, current)
    np.testing.assert_array_equal(original[:, 0], [1, 0])
    np.testing.assert_array_equal(tracked[:, 0], [0, 1])
    np.testing.assert_array_equal(tracked[:, 1:], current)


def test_camera_projection_can_associate_across_motion_and_one_missing_frame():
    tracker = MotionAssociation(False)
    move = np.array([[1., 0., 25.], [0., 1., 0.], [0., 0., 1.]])
    assert tracker.step(0, boxes(10))[0, 0] == 0
    assert tracker.step(1, boxes(), move).shape == (0, 7)
    assert tracker.step(2, boxes(60), move)[0, 0] == 0
    with pytest.raises(ValueError, match='every sampled frame'):
        tracker.step(4, boxes(90))


def test_two_frame_motion_recovery_and_empty_background_fallback():
    rng = np.random.default_rng(21)
    old = np.zeros((400, 600), dtype=np.uint8)
    for x, y in rng.integers([20, 20], [580, 380], size=(300, 2)):
        cv2.circle(old, (x, y), 3, 255, -1)
    affine = np.array([[1., 0., 14.], [0., 1., -9.]])
    new = cv2.warpAffine(old, affine, (600, 400))
    estimate, stats = camera_step(old, new, boxes(), boxes())
    assert stats['accepted']
    np.testing.assert_allclose(estimate[:2], affine, atol=.1)
    fallback, stats = camera_step(np.zeros_like(old), new, boxes(), boxes())
    assert not stats['accepted']
    np.testing.assert_array_equal(fallback, np.eye(3))


def test_reference_ceiling_does_not_require_unique_boxes_but_does_require_all_past():
    prefix = {'past': [{'detections': [[0, 5., 10., 15., 20.]]} for _ in range(8)]}
    label = [np.array([[1., 5., 10., 15., 20.], [2., 6., 10., 16., 20.]])] * 20
    assert possible_reference_support(prefix, label) == 2
    prefix['past'][4] = {'detections': []}
    assert possible_reference_support(prefix, label) == 0
