import numpy as np
import pytest

from src.world_model.m3w_source_box_motion import (
    annotation_displacement, regions, motion_tokens, fit_motion_normalizer, token_payload)


def test_crop_origin_and_anisotropic_resize_are_restored():
    flow = np.full((32, 32, 2), [-1, 2.])
    boxes = np.array([[38, 30, 58, 66], [41, 34, 61, 70.]])
    result = annotation_displacement(flow, boxes, np.array([.5, .25]))
    np.testing.assert_array_equal(result, np.broadcast_to([0, 40.], result.shape))


def test_background_region_excludes_box_and_rim():
    box = np.array([[38, 30, 58, 66.], [38, 30, 58, 66.]])
    inside, outside = regions(box)
    assert not (inside & outside).any()
    assert inside.sum() > 0 and outside.sum() > inside.sum()
    assert not outside[:, :2].any() and not outside[:, :, -2:].any()


def test_motion_frame_is_past_only_rotation_and_scale():
    raw = np.ones((2, 7, 19), np.float32)
    rotation = np.broadcast_to(np.eye(2), (2, 2, 2))
    result = motion_tokens(raw, rotation, np.ones(2)*2, np.ones(2)*3)
    np.testing.assert_allclose(result[..., :10], 1/6)
    np.testing.assert_array_equal(result[..., 10:], 1)


def test_quality_pair_is_exact_after_masking_motion():
    values = np.random.default_rng(4).normal(size=(10, 7, 19)).astype(np.float32)
    norm = fit_motion_normalizer(values[:6])
    quality = token_payload(values, norm, 'quality')
    motion = token_payload(values, norm, 'motion')
    np.testing.assert_array_equal(quality[..., 10:], motion[..., 10:])
    assert not quality[..., :10].any() and not motion[:, 0].any()
    assert not motion[..., 19:].any()
    frozen = [a.copy() for a in norm]
    values[6:] = 1e10
    token_payload(values, norm, 'motion')
    for a, b in zip(norm, frozen): np.testing.assert_array_equal(a, b)


def test_invalid_scales_and_role_rejected():
    with pytest.raises(ValueError):
        annotation_displacement(np.zeros((32, 32, 2)), np.ones((2, 4)), np.array([0, 1]))
    with pytest.raises(ValueError):
        token_payload(np.zeros((1, 7, 19)), (0, 1, []), 'future')
