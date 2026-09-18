import numpy as np
import pytest
from scripts.audit_m3w_source_site_quality import restored_label_extent, at_most_one_pixel, error_mass_slices


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


def test_label_count_is_not_error_mass_and_empty_error_is_explicit():
    values=error_mass_slices(np.array([0,1,1,10]),np.array([0,.01,.02,.5]))
    assert values[1]['window_fraction']==.5
    assert values[1]['cv_ade_error_mass_fraction']==pytest.approx(1/6)
    assert values[3]['cv_ade_error_mass_fraction']==pytest.approx(5/6)
    assert sum(v['rows'] for v in values)==4
    assert error_mass_slices(np.zeros(2),np.zeros(2))[0]['cv_ade_error_mass_fraction'] is None
    with pytest.raises(ValueError):
        error_mass_slices(np.array([-1]),np.array([.1]))
