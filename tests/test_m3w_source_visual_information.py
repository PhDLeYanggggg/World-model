import numpy as np
import pytest
from scripts.audit_m3w_source_visual_information import temporal_pixel_variation


def test_only_common_supported_pixels_contribute():
    rgb = np.zeros((2, 3, 3, 4, 4), np.uint8)
    cov = np.full((2, 3, 4, 4), 9, np.uint8)
    rgb[0, 1:] = 10
    cov[1] = 0
    rgb[1] = 255
    result = temporal_pixel_variation(rgb, cov)
    np.testing.assert_array_equal(result['mean_absolute_uint8_difference'], [[10, 0], [0, 0]])
    np.testing.assert_array_equal(result['identical_supported_pairs'], [[False, True], [False, False]])
    rgb[0, 2, :, 0, 0] = 255
    cov[0, 2, 0, 0] = 0
    result = temporal_pixel_variation(rgb, cov)
    assert result['mean_absolute_uint8_difference'][0, 1] == 0
    assert result['common_pixels'][0, 1] == 15


def test_pixel_statistics_reject_unaligned_coverage():
    with pytest.raises(ValueError, match='Aligned'):
        temporal_pixel_variation(np.zeros((2, 8, 3, 4, 4)), np.ones((2, 7, 4, 4)))
