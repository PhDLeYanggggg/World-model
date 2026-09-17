import json
import numpy as np
import pytest

from src.evaluation.m3w_past_video_alignment import native_to_image_xy,past_frame_indices,requested_crop_box


def test_axis_convention_is_explicit_and_swaps_only_output():
    q=np.array([[7.,19.]])
    np.testing.assert_array_equal(native_to_image_xy(q,np.eye(3),projected_axes='row_col'),[[19,7]])
    np.testing.assert_array_equal(native_to_image_xy(q,np.eye(3),projected_axes='xy'),q)
    with pytest.raises(ValueError):
        native_to_image_xy(q,np.eye(3),projected_axes='guess')


def test_past_index_requests_do_not_rescale_using_video_fps():
    t=past_frame_indices(2904,2904+np.arange(-7,1)*6)
    np.testing.assert_array_equal(t,np.arange(2862,2905,6))
    with pytest.raises(ValueError):
        past_frame_indices(2904,np.arange(2862,2905,6)+1)


@pytest.mark.parametrize('times', [[-1,0,1,2,3,4,5,6],[1,2,3,4,5,6,7,7],[1,2,3,4,5,6,7.5,8],[1,2]])
def test_invalid_history_rejected(times):
    with pytest.raises(ValueError):
        past_frame_indices(times[-1],times)


def test_crop_bounds_mark_clipping_without_inventing_body_detection():
    full=requested_crop_box([100,100],640,480)
    assert full['box']==(52,20,148,116) and full['full_support']
    edge=requested_crop_box([1,1],640,480)
    assert not edge['full_support'] and edge['point_inside']
    assert requested_crop_box([-200,-200],640,480) is None
    assert json.loads(json.dumps(full))['box']==[52,20,148,116]
