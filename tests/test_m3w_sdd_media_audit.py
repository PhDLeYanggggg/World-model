import numpy as np
import pytest

from src.world_model.m3w_sdd_media_audit import (
    annotation_summary, decoded_coverage, reference_comparison,
)


def sample():
    return np.array([[0, 1, 1, 5, 8, 0, 0, 0, 0],
                     [0, 2, 1, 6, 8, 1, 0, 1, 1]], dtype=float)


def test_source_summary_keeps_interpolation_and_visibility():
    r = annotation_summary(sample(), 10, 10)
    assert r['frame_union_contiguous']
    assert r['generated_rows'] == 1 and r['occluded_rows'] == 1
    assert r['duplicate_agent_frame_rows'] == 0


def test_duplicate_keys_and_outside_boxes_are_reported_not_silently_dropped():
    a = np.vstack([sample(), sample()[:1]])
    a[0, 1] = -1
    r = annotation_summary(a, 10, 10)
    assert r['duplicate_agent_frame_rows'] == 1
    assert r['visible_boxes_outside_image'] == 1 and r['rows'] == 3


@pytest.mark.parametrize('column,value', [(5, .5), (5, -1), (6, 2), (8, float('nan'))])
def test_invalid_frame_or_flags_rejected(column, value):
    a = sample()
    a[0, column] = value
    with pytest.raises(ValueError):
        annotation_summary(a, 10, 10)


def test_decoder_boundary_is_zero_based_and_not_header_assumption():
    r = decoded_coverage(np.array([0, 1, 2, 2]), 2, 3)
    assert r['annotation_rows_outside_decode'] == 2
    assert not r['header_matches_decode']
    assert not r['decoded_range_covers_annotations']


def test_complete_frame_range_is_not_semantic_or_physical_certification():
    r = decoded_coverage(np.array([0, 1]), 2, 2)
    assert r['decoded_range_covers_annotations']
    assert not r['semantic_image_annotation_alignment_certified']
    assert not r['physical_time_or_scale_certified']


def test_reference_shape_mismatch_is_retained():
    r = reference_comparison(np.zeros((3, 3, 3)), np.zeros((4, 3, 3)))
    assert not r['shape_match']


def test_exact_reference_still_does_not_certify_all_frames():
    a = np.arange(27).reshape(3, 3, 3)
    r = reference_comparison(a, a)
    assert r['exact_pixels'] and r['pixel_correlation'] == pytest.approx(1)
    assert not r['semantic_alignment_certified']
