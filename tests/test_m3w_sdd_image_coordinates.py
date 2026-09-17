import numpy as np
import pytest

from src.world_model.m3w_sdd_image_coordinates import SDDImageCoordinates
from src.world_model.m3w_masked_history_images import masked_center_patch


def test_reference_edges_map_to_video_without_mutating_state():
    mapper = SDDImageCoordinates(100, 200, 50, 80)
    original = np.array([[0., 0., 100., 200.], [-2., 3., 8., 13.]])
    saved = original.copy()
    mapped = mapper.past_boxes(original, frame_ids=[0, 7], query_frame=7)
    np.testing.assert_array_equal(original, saved)
    np.testing.assert_allclose(mapped[0], [0, 0, 50, 80])
    np.testing.assert_allclose(mapper.video_boxes_to_annotation(mapped), original)
    assert mapped[1, 0] < 0  # Partial support is preserved, not silently clipped.


def test_future_annotation_is_rejected_by_crop_api():
    mapper = SDDImageCoordinates(100, 100, 50, 50)
    with pytest.raises(ValueError):
        mapper.past_boxes([[1, 1, 5, 5]], frame_ids=[8], query_frame=7)


@pytest.mark.parametrize('frame', [-1, .5, float('nan')])
def test_invalid_frame_rejected(frame):
    with pytest.raises(ValueError):
        SDDImageCoordinates(2, 2, 2, 2).past_boxes([[0, 0, 1, 1]], frame_ids=[frame], query_frame=1)


@pytest.mark.parametrize('sizes', [(0, 2, 2, 2), (2, -2, 2, 2), (2.5, 2, 2, 2), (True, 2, 2, 2)])
def test_invalid_dimensions_rejected(sizes):
    with pytest.raises(ValueError):
        SDDImageCoordinates(*sizes)


def test_identity_is_exact_and_does_not_claim_metric():
    mapper = SDDImageCoordinates(200, 100, 200, 100)
    box = np.array([[5., 10., 20., 35.]])
    np.testing.assert_array_equal(mapper.past_boxes(box, frame_ids=[0], query_frame=0), box)
    assert not mapper.metadata()['homography_or_metric_calibration']


def test_corrected_observed_crop_hits_synthetic_target_in_resized_video():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[26:35, 16:25] = 255
    native_box = np.array([[32., 52., 48., 68.]])
    mapped = SDDImageCoordinates(200, 200, 100, 100).past_boxes(
        native_box, frame_ids=[0], query_frame=0)
    current_center = (mapped[0, :2]+mapped[0, 2:])/2
    old_center = (native_box[0, :2]+native_box[0, 2:])/2
    good, mask = masked_center_patch(image, current_center, 16, 16)
    wrong, _ = masked_center_patch(image, old_center, 16, 16)
    assert good.sum() > 0 and mask.all()
    assert not wrong.any()
