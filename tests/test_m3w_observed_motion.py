import sys
from pathlib import Path
import numpy as np
import pytest

from src.world_model.m3w_observed_motion import (
    feature_variant, full_image_correspondence, image_to_native, normalized_motion,
    pair_motion, validate_past_join,
)


def test_recentered_static_background_is_not_object_motion():
    flow = np.zeros((32, 32, 2), np.float32)
    flow[..., 0], flow[..., 1] = -2, 1
    source, destination = full_image_correspondence(flow, [100, 200], [106, 197])
    np.testing.assert_array_equal(source, destination)
    np.testing.assert_array_equal(source[0, 0], [53, 153])


def test_axes_and_heading_rotation_not_raw_camera_motion():
    xy = np.array([[10., 20.], [12., 23.]])
    native, valid = image_to_native(xy, np.diag([2., 3., 1.]))
    np.testing.assert_array_equal(native, [[40, 30], [46, 36]])
    assert valid.all()
    x = normalized_motion(np.tile([6., 6.], (3, 1)), np.ones(4), [[0., -1.], [1., 0.]], 2.)
    np.testing.assert_array_equal(x[:6], [3, -3, 3, -3, 3, -3])
    assert np.all(x[6:] == .5)


def test_past_join_rejects_future_and_wrong_identity():
    history = np.column_stack([np.arange(8), np.ones(8), np.zeros((8, 2))])
    assert validate_past_join(history, np.arange(8), {'agent': 1, 'frame': 7}, np.arange(8))
    with pytest.raises(ValueError):
        validate_past_join(history, np.arange(8), {'agent': 1, 'frame': 6}, np.arange(8))
    with pytest.raises(ValueError):
        validate_past_join(history, np.arange(8), {'agent': 2, 'frame': 7}, np.arange(8))
    with pytest.raises(ValueError):
        validate_past_join(history, np.arange(8), {'agent': 1, 'frame': 7}, np.arange(8)+1)


def test_variants_match_quality_dimension_and_censor_direction():
    geometry, motion, quality = np.ones((2, 3)), np.ones((2, 7, 10)), np.ones((2, 7, 5))*2
    full = feature_variant(geometry, motion, quality, 'directed')
    control = feature_variant(geometry, motion, quality, 'quality_control')
    magnitude = feature_variant(geometry, motion, quality, 'magnitude')
    assert full.shape == control.shape == magnitude.shape == (2, 108)
    np.testing.assert_array_equal(full[:, :38], control[:, :38])
    assert not control[:, 38:].any()
    assert not magnitude[:, 38:].reshape(2, 7, 10)[:, :, :6].any()
    assert np.all(magnitude[:, 38:].reshape(2, 7, 10)[:, :, 6:] == 1)
    assert np.all(motion == 1)


def test_real_flow_on_synthetic_translation_and_empty_masks():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'data/stage_cvpr2027_experiments/optical_flow_runtime'))
    import cv2
    cv2.setNumThreads(1)
    rng = np.random.default_rng(41)
    gray = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), .5)
    first = np.repeat(gray[None], 3, axis=0)
    second = np.repeat(cv2.warpAffine(gray, np.float32([[1, 0, 1], [0, 1, 0]]), (32, 32))[None], 3, axis=0)
    coverage = np.ones((32, 32), np.uint8)*9
    vectors, _, quality = pair_motion(first, second, coverage, coverage, [100, 100], [100, 100], np.eye(3), cv2)
    np.testing.assert_allclose(vectors[0], [0, 3], atol=.25)
    assert quality[2] > .8
    corrected, _, _ = pair_motion(first, second, coverage, coverage, [100, 100], [97, 100], np.eye(3), cv2)
    np.testing.assert_allclose(corrected[0], [0, 0], atol=.25)
    vec, mag, quality = pair_motion(first, second, coverage*0, coverage, [100, 100], [100, 100], np.eye(3), cv2)
    assert not vec.any() and not mag.any() and not quality.any()
