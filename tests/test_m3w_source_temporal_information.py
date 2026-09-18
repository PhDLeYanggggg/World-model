import numpy as np
import pytest
from src.evaluation.m3w_source_temporal_information import verify_history_keys, box_masks, temporal_diagnostics


def test_history_keys_reject_future_repeats_and_other_agent():
    q = np.array([[96, 7]]); keys = np.column_stack((np.arange(12, 97, 12), np.full(8, 7)))
    rows = np.arange(8)[None]
    np.testing.assert_array_equal(verify_history_keys(q, keys, rows), keys[None])
    for change in ('frame', 'agent', 'repeat'):
        bad = keys.copy()
        if change == 'frame': bad[-1, 0] += 12
        elif change == 'agent': bad[-1, 1] += 1
        else: bad[-1] = bad[-2]
        with pytest.raises(ValueError): verify_history_keys(q, bad, rows)


def test_box_footprint_uses_actual_centered_pixel_coordinates():
    masks = box_masks(np.array([[45,45,51,51], [0,0,96,96]], float))
    assert masks[0].sum() == 4 and masks[1].all()


def inputs():
    rgb = np.zeros((1,8,3,32,32), np.uint8)
    cov = np.full((1,8,32,32), 9, np.uint8)
    boxes = np.tile([45,45,51,51], (1,8,1)).astype(float)
    emb = np.zeros((1,8,512), np.float32); emb[..., 0] = 1
    return rgb, cov, boxes, emb


def test_temporal_change_regions_and_constant_embeddings():
    rgb, cov, boxes, emb = inputs()
    rgb[0,1:, :,15:17,15:17] = 7
    v = temporal_diagnostics(rgb, cov, boxes, emb)
    assert v['identical_adjacent_pairs'][0] == 6
    assert v['box_pixel_change'][0] == 1
    assert v['outside_box_pixel_change'][0] == 0
    assert v['temporal_embedding_energy_fraction'][0] == 0
    assert v['identical_embedding_sequence'][0]


def test_absent_pixels_are_missing_not_zero_motion():
    rgb, cov, boxes, emb = inputs(); cov[:] = 0; emb[:] = 0
    v = temporal_diagnostics(rgb, cov, boxes, emb)
    assert v['valid_frames'][0] == 0 and np.isnan(v['past_pixel_change'][0])
    assert v['common_pixel_pairs'][0] == 0


def test_nonconstant_embedding_energy_and_invalid_shapes():
    rgb, cov, boxes, emb = inputs(); emb[:,4:,0] = 0; emb[:,4:,1] = 1
    v = temporal_diagnostics(rgb, cov, boxes, emb)
    assert np.isclose(v['temporal_embedding_energy_fraction'][0], .5)
    with pytest.raises(ValueError): temporal_diagnostics(rgb[:,:7], cov, boxes, emb)
