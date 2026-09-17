import sys
from pathlib import Path
import numpy as np
import pytest
from src.world_model.m3w_spatial_motion import reduce_patch, grid_pool, dense_pair, feature_variant
from src.world_model.m3w_masked_history_images import masked_center_patch


def test_raw_patch_reduction_replays_masked_source_exactly():
    image = np.random.default_rng(19).integers(0, 256, (118, 171, 3), dtype=np.uint8)
    for center in ([80, 59], [0, 0], [169, 115], [-200, 50]):
        native, mask = masked_center_patch(image, center, 96, 96)
        small, count = reduce_patch(native, mask)
        expected, expected_count = masked_center_patch(image, center, 96, 32)
        np.testing.assert_array_equal(small, expected)
        np.testing.assert_array_equal(count, expected_count)


def test_sparse_motion_survives_grid_but_not_global_median():
    flow = np.zeros((96, 96, 2))
    flow[25:40, 25:40, 0] = 4.
    grid, quality, pooled = grid_pool(flow, np.ones((96, 96), bool), np.ones((96, 96)))
    assert pooled[0] == 0 and grid[5, 0] > 1
    assert not grid[0].any()
    assert np.all(quality == 1)


def test_real_native_flow_compensates_crop_motion():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'data/stage_cvpr2027_experiments/optical_flow_runtime'))
    import cv2
    cv2.setNumThreads(1)
    gray = np.random.default_rng(22).integers(0, 256, (96, 96), dtype=np.uint8)
    first = np.repeat(gray[None], 3, 0)
    second = np.repeat(cv2.warpAffine(gray, np.float32([[1,0,2],[0,1,0]]), (96,96))[None], 3, 0)
    mask = np.ones((96,96), np.uint8)
    grid, _, pooled = dense_pair(first, second, mask, mask, [[100,100],[98,100]], np.eye(3), cv2)
    np.testing.assert_allclose(grid[5, :2], [0,0], atol=.2)
    np.testing.assert_allclose(pooled[:2], [0,0], atol=.2)
    empty, support, _ = dense_pair(first, second, mask*0, mask, [[100,100],[98,100]], np.eye(3), cv2)
    assert not empty.any() and not support.any()


def test_control_information_and_model_dimensions_match():
    geometry = np.ones((2, 4))
    motion = np.zeros((2,3,7,16,3))
    for i in range(3):
        motion[:, i] = i+1
    quality = np.ones((2,2,7,16,2))*4
    for arm, expected in [('quality_control',0),('lowpass_pool',3),('lowpass_grid',1),('native_grid',2)]:
        x = feature_variant(geometry, motion, quality, arm)
        assert x.shape == (2, 788)
        assert np.all(x[:,4:452] == 4)
        assert np.all(x[:,452:] == expected)
    with pytest.raises(ValueError):
        feature_variant(geometry, motion, quality, 'selected_after_test')
