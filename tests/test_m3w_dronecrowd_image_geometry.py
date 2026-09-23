"""Synthetic registration checks, not real-scene independence evidence."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from src.evaluation.m3w_dronecrowd_image_geometry import (
    background_features, candidate_components, match_background,
)


def texture(seed):
    rng = np.random.default_rng(seed)
    image = np.full((1080,1920,3),220,np.uint8)
    for _ in range(700):
        x,y = rng.integers(20,1900),rng.integers(20,1060)
        cv2.circle(image,(int(x),int(y)),int(rng.integers(3,15)),
                   tuple(map(int,rng.integers(0,180,3))),-1)
    return image


def test_known_pixel_translation_recovers_only_image_transform():
    cv2.setNumThreads(1)
    image = texture(4)
    moved = cv2.warpAffine(image,np.array([[1,0,24],[0,1,12]],float),(1920,1080))
    result = match_background(background_features(image,[]),background_features(moved,[]))
    assert result["overlap_candidate"]
    assert result["inliers"]>100
    assert result["median_grid_displacement_pixels"] == pytest.approx(np.hypot(24,12),abs=1)
    assert result["transform_kind"]=="image_to_image_only"


def test_head_mask_and_no_features_do_not_manufacture_match():
    image = texture(5)
    blank = background_features(image,[[0,0,1920,1080]])
    assert blank["background_fraction"]==0
    assert blank["descriptors"] is None
    assert not match_background(blank,background_features(image,[]))["overlap_candidate"]


def test_unrelated_synthetic_scenes_do_not_match():
    result = match_background(background_features(texture(1),[]),background_features(texture(2),[]))
    assert not result["overlap_candidate"]


def test_out_of_bounds_box_cannot_mask_opposite_side_of_image():
    item = background_features(texture(3),[[-100,-100,-90,-90],[2000,1200,2010,1210]])
    assert item["background_fraction"]==1


def test_components_keep_uncertain_singletons_without_independence_claim():
    assert candidate_components(["a","b","c","d"],[("a","b"),("b","c")]) == [["a","b","c"],["d"]]
