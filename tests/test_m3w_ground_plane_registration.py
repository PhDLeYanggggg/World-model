import cv2
import numpy as np

from src.evaluation.m3w_ground_plane_registration import grid_to_utm, holdouts, project


def test_pixel_center_and_y_flip():
    np.testing.assert_allclose(grid_to_utm([[0,0],[9,9]],(100,200,110,210),(10,10)),
                               [[100.5,209.5],[109.5,200.5]])
    np.testing.assert_allclose(project([[1,2]], np.array([[2,0,3],[0,2,4],[0,0,1]])), [[5,8]])


def test_held_patch_detects_small_translation():
    rng = np.random.default_rng(7)
    a = rng.integers(0,255,size=(150,150),dtype=np.uint8)
    b = cv2.warpAffine(a,np.array([[1,0,3],[0,1,-2]],dtype=float),a.shape[::-1])
    check = holdouts(a,b,np.eye(3),{"center":(75,75)})[0]
    assert check["delta_pixels"] == [3,-2] and check["peak"] > .99
