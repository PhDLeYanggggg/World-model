import numpy as np
import pytest

from src.world_model.m3w_continuation_boxes import ContinuationBoxes


def boxes(*centers):
    return np.asarray([[x-5, 0., x+5, 10., .9, 1.] for x in centers]).reshape(-1, 6)


@pytest.mark.parametrize('one_gap', (True, False))
def test_fine_boxes_only_extend_tracks_and_do_not_compete_with_coarse(one_gap):
    tracker = ContinuationBoxes(one_gap)
    primary, fine = tracker.step(0, boxes(10, 30), boxes(100))
    assert primary[:, 0].tolist() == [0, 1]
    assert fine.shape == (0, 7)
    primary, fine = tracker.step(1, boxes(10), boxes(10, 30, 100))
    assert primary[:, 0].tolist() == [0]
    assert fine[:, 0].tolist() == [1]
    np.testing.assert_array_equal(fine[:, 1:], boxes(30))
    primary, fine = tracker.step(2, boxes(10), boxes(30, 100))
    assert fine[:, 0].tolist() == ([] if one_gap else [1])
    primary, _ = tracker.step(3, boxes(10, 30, 100), boxes())
    assert primary[:, 0].tolist() == [0, 1, 2]


def test_extension_uses_current_camera_transform_and_never_revives_expired_track():
    tracker = ContinuationBoxes(False)
    move = np.array([[1., 0., 30.], [0., 1., 0.], [0., 0., 1.]])
    tracker.step(0, boxes(10), boxes())
    assert tracker.step(1, boxes(), boxes(40), move)[1][:, 0].tolist() == [0]
    tracker.step(2, boxes(), boxes(), move)
    tracker.step(3, boxes(), boxes(), move)
    assert tracker.step(4, boxes(), boxes(130), move)[1].shape == (0, 7)


def test_deduplication_cannot_use_a_coarse_box_to_update_two_tracks():
    tracker = ContinuationBoxes(False)
    tracker.step(0, boxes(5, 10), boxes())
    primary, fine = tracker.step(1, boxes(5), boxes(6))
    assert primary[:, 0].tolist() == [0]
    assert fine.shape == (0, 7)
