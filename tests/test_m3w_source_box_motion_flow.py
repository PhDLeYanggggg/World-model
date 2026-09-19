from pathlib import Path
import sys

import numpy as np
import pytest

from src.world_model.m3w_source_box_motion import pair_features


@pytest.fixture
def cv():
    runtime = Path(__file__).resolve().parents[1]/'data/stage_cvpr2027_experiments/optical_flow_runtime'
    if not runtime.exists(): pytest.skip('Optional bound OpenCV runtime not installed')
    sys.path.insert(0, str(runtime))
    import cv2
    cv2.setNumThreads(1)
    return cv2


def test_identical_crops_restore_translation_without_fake_contrast(cv):
    image = np.random.default_rng(17).integers(0, 256, (3, 32, 32), dtype=np.uint8)
    boxes = np.array([[30, 30, 66, 66], [33, 36, 69, 72.]])
    value = pair_features(np.stack([image, image]), np.full((2, 32, 32), 9, np.uint8),
                          boxes, np.zeros((2, 3), np.uint8), np.array([.5, .25]), cv)
    np.testing.assert_allclose(value[:4], [6, 24, 6, 24], atol=.05)
    np.testing.assert_allclose(value[4:6], 0, atol=.05)
    np.testing.assert_array_equal(value[14:16], 1)


def test_missing_image_and_tiny_box_stay_supported_by_explicit_flags(cv):
    images = np.zeros((2, 3, 32, 32), np.uint8)
    boxes = np.array([[47, 47, 49, 49], [47, 47, 49, 49.]])
    flags = np.array([[0, 1, 1], [0, 0, 1]], np.uint8)
    absent = pair_features(images, np.zeros((2, 32, 32), np.uint8), boxes, flags, np.ones(2), cv)
    assert not absent[:16].any()
    np.testing.assert_array_equal(absent[16:], [0, 1, 1])
    tiny = pair_features(images, np.full((2, 32, 32), 9, np.uint8), boxes, flags, np.ones(2), cv)
    assert tiny[14] == 0 and tiny[15] == 1
    assert not tiny[:2].any() and not tiny[4:6].any()
