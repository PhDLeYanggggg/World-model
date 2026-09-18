import numpy as np
import pytest
from scripts.audit_m3w_source_site_quality import restored_label_extent, at_most_one_pixel


def test_label_extent_restores_pixels_and_uses_past_box_only():
    target=np.zeros((2,12,2));target[0,5]=[3,4]
    pixels,ratio=restored_label_extent(target,np.array([2,3]),np.array([[0,0,6,8],[5,5,8,9]]))
    np.testing.assert_array_equal(pixels,[10,0]);np.testing.assert_array_equal(ratio,[1,0])


def test_invalid_box_or_scale_is_not_guessed():
    with pytest.raises(ValueError):
        restored_label_extent(np.zeros((1,12,2)),np.array([0]),np.array([[0,0,1,1]]))
    with pytest.raises(ValueError):
        restored_label_extent(np.zeros((1,12,2)),np.array([1]),np.array([[0,0,0,0]]))


def test_one_pixel_bin_handles_cached_float32_roundoff_not_larger_motion():
    np.testing.assert_array_equal(at_most_one_pixel([.5,1,1.000000047,1.0001,1.5]),
                                  [True,True,True,False,False])
