import numpy as np

from src.evaluation.m3w_movingdronecrowd_orientation import canonical_rgb_boxes


def test_landscape_input_and_boxes_remain_exactly_unchanged():
    image=np.random.default_rng(7).integers(0,256,(18,32,3),dtype=np.uint8)
    boxes=np.array([[3.,4.,9.,12.]])
    actual,rectangles,t,rotation=canonical_rgb_boxes(image,boxes)
    np.testing.assert_array_equal(actual,image);np.testing.assert_array_equal(rectangles,boxes)
    np.testing.assert_array_equal(t,np.eye(3));assert rotation==0


def test_portrait_rotation_preserves_pixels_box_centers_and_canonical_aspect():
    original=np.random.default_rng(8).integers(0,256,(18,32,3),dtype=np.uint8)
    box=np.array([[3.,4.,9.,12.]])
    portrait=np.rot90(original,k=-1).copy()
    portrait_box=np.column_stack((18-1-box[:,3],box[:,0],18-1-box[:,1],box[:,2]))
    actual,rectangles,t,rotation=canonical_rgb_boxes(portrait,portrait_box)
    np.testing.assert_array_equal(actual,original);np.testing.assert_array_equal(rectangles,box)
    center=np.r_[(portrait_box[0,:2]+portrait_box[0,2:])/2,1.]
    np.testing.assert_array_equal((t@center)[:2],(box[0,:2]+box[0,2:])/2)
    assert rotation==90 and actual.shape[1]/actual.shape[0]==16/9


def test_empty_head_inventory_is_preserved_under_portrait_rotation():
    actual,boxes,_,rotation=canonical_rgb_boxes(np.zeros((32,18,3),np.uint8),np.empty((0,4)))
    assert actual.shape==(18,32,3) and boxes.shape==(0,4) and rotation==90
