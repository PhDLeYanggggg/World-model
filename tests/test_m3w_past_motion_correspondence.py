import json
import numpy as np
import pytest

from src.evaluation.m3w_past_motion_correspondence import (
    centered_patch, match_past_patch, select_moving_controls,
)


def test_control_selection_uses_first_complete_moving_past():
    points = np.c_[np.arange(20)*6, np.ones(20), np.arange(20)*2, np.zeros(20)]
    first, count = select_moving_controls(points, np.eye(3), 'fit')
    assert count == 1 and first[0]['frame_id'] == 42
    assert first[0]['history_frames'] == list(range(0, 43, 6))
    changed = points.copy()
    changed[8:, 2:4] = 1e7
    assert select_moving_controls(changed, np.eye(3), 'fit') == (first, count)
    assert json.loads(json.dumps(first)) == first


def test_control_selection_rejects_broken_history():
    points = np.c_[np.arange(8)*6, np.ones(8), np.arange(8)*2, np.zeros(8)]
    points[3, 0] += 1
    assert select_moving_controls(points, np.eye(3), 'fit') == ([], 0)


def test_centered_crop_does_not_assume_footpoint_or_fill_edges():
    a = np.arange(100*100).reshape(100, 100)
    np.testing.assert_array_equal(centered_patch(a, [50, 50], 5), a[48:53, 48:53])
    assert centered_patch(a, [1, 1], 15) is None


def test_zncc_recovers_translation_without_later_annotation():
    a = np.random.default_rng(17).normal(size=(100, 100))
    b = np.roll(a, (3, -5), axis=(0, 1)) * 2 + 4
    result = match_past_patch(a, b, [50, 50], template_size=15, search_radius=10)
    assert result['status'] == 'matched'
    np.testing.assert_array_equal(result['image_xy'], [45, 53])
    assert result['peak_zncc'] == pytest.approx(1)
    assert json.loads(json.dumps(result)) == result


def test_flat_and_missing_support_are_not_silent_zero_motion():
    a = np.zeros((100, 100))
    assert match_past_patch(a, a, [50, 50])['status'] == 'flat_template'
    assert match_past_patch(a, a, [1, 1])['status'] == 'out_of_image_support'
    with pytest.raises(ValueError):
        match_past_patch(a, a, [50, 50], template_size=16)
