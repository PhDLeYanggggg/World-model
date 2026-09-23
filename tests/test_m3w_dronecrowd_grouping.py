import json
from pathlib import Path

import numpy as np
import pytest
cv2 = pytest.importorskip("cv2")

from src.evaluation.m3w_dronecrowd_grouping import (
    compare_clip_views,extract_features,group_evidence,pair_evidence,require_no_cross_role_edges,
)

CONFIG = json.loads((Path(__file__).resolve().parents[1]/"configs/m3w_dronecrowd_grouping_v2.json").read_text())


def texture(seed):
    rng = np.random.default_rng(seed)
    image = np.full((1080,1920,3),210,np.uint8)
    for _ in range(800):
        x,y = rng.integers(20,1900),rng.integers(20,1060)
        cv2.circle(image,(int(x),int(y)),int(rng.integers(3,14)),tuple(map(int,rng.integers(0,180,3))),-1)
    return image


def test_rotation_brightness_partial_view_is_supported():
    cv2.setNumThreads(1)
    image = texture(29)
    matrix = cv2.getRotationMatrix2D((960,540),23,0.85)
    moved = cv2.warpAffine(image,matrix,(1920,1080))
    moved = np.clip(moved.astype(float)*0.55+15,0,255).astype(np.uint8)
    a,b = extract_features(image,[],CONFIG),extract_features(moved,[],CONFIG)
    result = pair_evidence(a,b,CONFIG)
    assert result["status"]=="strong"
    assert result["inliers"]>60


def test_unrelated_or_masked_source_is_not_declared_independent():
    a,b = extract_features(texture(3),[],CONFIG),extract_features(texture(6),[],CONFIG)
    assert pair_evidence(a,b,CONFIG)["status"]=="unsupported"
    blank = extract_features(texture(3),[[0,0,1920,1080]],CONFIG)
    assert pair_evidence(a,blank,CONFIG)["status"]=="unsupported"
    result = group_evidence(["a","b"],[])
    assert not result["negative_matches_establish_independence"]


def test_ambiguous_edge_must_not_be_split_between_roles():
    pairs = [{"left":"a","right":"b","status":"strong_multiview"},
             {"left":"b","right":"c","status":"ambiguous"}]
    result = group_evidence(["a","b","c","d"],pairs)
    assert result["strong_components"]==[["a","b"],["c"],["d"]]
    assert result["conservative_components"]==[["a","b","c"],["d"]]
    with pytest.raises(ValueError,match="crosses"):
        require_no_cross_role_edges(dict(a="fit",b="fit",c="test"),pairs)
    require_no_cross_role_edges(dict(a="fit",b="fit",c="excluded"),pairs)


def test_prior_positive_evidence_is_not_erased_by_negative_rematch():
    result = group_evidence(["a","b"],[],[("a","b")])
    assert result["conservative_components"]==[["a","b"]]
    with pytest.raises(ValueError,match="crosses"):
        require_no_cross_role_edges(dict(a="fit",b="test"),[],[("a","b")])


def test_multiview_requires_distinct_frames_on_both_sides(monkeypatch):
    import src.evaluation.m3w_dronecrowd_grouping as module
    def only_one_left(left,right,config):
        return {"status":"strong" if left==1 else "unsupported"}
    monkeypatch.setattr(module,"pair_evidence",only_one_left)
    views = {f:f for f in CONFIG["selected_frames_one_based"]}
    assert compare_clip_views(views,views,CONFIG)["status"]=="ambiguous"


def test_early_exit_after_two_distinct_strong_pairs(monkeypatch):
    import src.evaluation.m3w_dronecrowd_grouping as module
    monkeypatch.setattr(module,"pair_evidence",lambda *args: {"status":"strong"})
    views = {f:f for f in CONFIG["selected_frames_one_based"]}
    result = compare_clip_views(views,views,CONFIG)
    assert result["status"]=="strong_multiview" and result["frame_pairs_attempted"]==2


def test_unknown_ids_cannot_enter_graph():
    with pytest.raises(ValueError):
        group_evidence(["a"],[],[("a","missing")])
