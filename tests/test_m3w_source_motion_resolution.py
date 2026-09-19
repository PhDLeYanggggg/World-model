import sys
from pathlib import Path

import numpy as np
import pytest

from src.world_model.m3w_source_motion_resolution import pair_features, displacement, raw_labels, VARIANTS
from src.world_model.m3w_source_box_motion import pair_features as old_pair
from src.world_model.m3w_sdd_past_images import supported_center_patch
from src.world_model.m3w_spatial_motion import reduce_patch


def cv():
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'data/stage_cvpr2027_experiments/optical_flow_runtime'))
    import cv2
    cv2.setNumThreads(1)
    return cv2


def test_native_reduction_matches_old_patch_with_boundary_and_mask():
    image=np.random.default_rng(7).integers(0,256,(120,130,3),dtype=np.uint8)
    mask=np.zeros((120,130),bool); mask[:23,:19]=True
    for center in ([4.5,8.5],[60.1,64.7],[128,116]):
        native=supported_center_patch(image,center,mask,96,96)
        old=supported_center_patch(image,center,mask,96,32)
        rgb,cov=reduce_patch(native['rgb_retained'],native['retained_count'])
        np.testing.assert_array_equal(rgb,old['rgb_retained'])
        np.testing.assert_array_equal(cov,old['retained_count'])


def test_old_control_is_bit_exact():
    rng=np.random.default_rng(11)
    first=rng.integers(0,256,(3,32,32),dtype=np.uint8)
    rgb=np.stack([first,np.roll(first,1,axis=2)])
    coverage=np.full((2,32,32),9,np.uint8)
    boxes=np.array([[80,70,110,112],[83,69,113,111]],float)
    flags=np.array([[0,0,1],[0,1,1]],np.uint8); scale=np.array([.8,1.1])
    np.testing.assert_array_equal(pair_features(rgb,coverage,boxes,flags,scale,cv(),'lowpass_w45'),
                                  old_pair(rgb,coverage,boxes,flags,scale,cv()))


@pytest.mark.parametrize('variant',list(VARIANTS))
def test_identical_frames_zero_motion(variant):
    size,_=VARIANTS[variant]; area=(96//size)**2
    rgb=np.zeros((2,3,size,size),np.uint8); coverage=np.full((2,size,size),area,np.uint8)
    result=pair_features(rgb,coverage,np.array([[30,30,65,65]]*2),np.zeros((2,3),np.uint8),[1,2],cv(),variant)
    np.testing.assert_array_equal(result[:10],np.zeros(10)); assert result[14]==1 and result[15]==1


def test_coordinate_restoration_is_anisotropic_and_causal():
    boxes=np.array([[0,0,30,40],[3,-2,33,38]])
    flow=np.ones((96,96,2),np.float32)
    np.testing.assert_allclose(displacement(flow,boxes,[2,.5]),np.broadcast_to([2,-2],flow.shape))
    with pytest.raises(ValueError): displacement(flow,boxes,[0,1])


def test_exact_raw_label_boundary():
    target=np.zeros((4,12,2)); target[0,:,0]=10; target[1,:,0]=10.000001
    target[2,:,:]=[6,8]
    np.testing.assert_array_equal(raw_labels(target,'over10_annotation_pixels'),[0,1,0,0])
    np.testing.assert_array_equal(raw_labels(target,'any_nonzero'),[1,1,1,0])
