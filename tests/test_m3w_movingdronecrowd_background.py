import cv2
import json
import numpy as np
import pytest

from src.evaluation.m3w_dronecrowd_grouping import extract_features
from src.evaluation.m3w_movingdronecrowd_background import (
    selected_frames,moving_node,frame_boxes,prepare_raster,features_from_raster,
    exact_features,pair_roster,conservative_exclusions,
)


def config():
    return dict(resize=[160,120],clahe_clip_limit=2.,max_features=100)


def test_three_distinct_view_slots_do_not_change_actual_frame_ids():
    assert selected_frames(range(1,301))==[1,150,300]
    assert selected_frames(range(1,11))==[1,5,10]
    with pytest.raises(ValueError):selected_frames([1,2])


def test_annotation_zero_frame_is_image_one_not_future_boxes():
    raw=b'0,1,1,2,3,4,-1,-1,-1,-1\n1,1,100,200,30,40,-1,-1,-1,-1'
    np.testing.assert_array_equal(frame_boxes(raw,0),[[1,2,4,6]])
    assert moving_node('scene_1/2')=='moving_scene_1__2'
    with pytest.raises(ValueError):moving_node('../scene_1/2')


def test_lossless_preparation_reproduces_original_rgb_rootSIFT_exactly():
    cv2.setNumThreads(1)
    rgb=np.random.default_rng(42).integers(0,256,(240,320,3),dtype=np.uint8)
    boxes=np.array([[10,10,60,80],[100,90,130,120]],float)
    gray,mask=prepare_raster(rgb,boxes,config())
    assert set(np.unique(mask))=={0,255}
    exact_features(extract_features(rgb,boxes,config()),features_from_raster(gray,mask,config()))


def test_constant_raster_preserves_empty_descriptor_semantics():
    gray,mask=prepare_raster(np.zeros((120,160,3),np.uint8),np.empty((0,4)),config())
    value=features_from_raster(gray,mask,config())
    assert value['xy'].shape==(0,2) and value['descriptors'] is None


def test_single_file_view_checkpoint_binds_its_pixels_and_source_identity(tmp_path):
    from scripts.audit_m3w_movingdronecrowd_background import read_view
    from src.evaluation.m3w_movingdronecrowd_background import raster_hash
    gray,mask=prepare_raster(np.zeros((120,160,3),np.uint8),np.empty((0,4)),config())
    value=features_from_raster(gray,mask,config())
    meta=dict(node='moving_scene_1__1',slot=1,raster_sha256=raster_hash(gray,mask))
    path=tmp_path/'view.npz'
    np.savez_compressed(path,gray=gray,mask=mask,xy=value['xy'],descriptors=np.empty((0,128),np.float32),
                        background_fraction=value['background_fraction'],metadata_json=np.asarray(json.dumps(meta)))
    read_view(path,dict(node='moving_scene_1__1',slot=1),config(),reextract=True)
    with pytest.raises(ValueError):read_view(path,dict(node='moving_scene_2__1'),config())
    gray[0,0]=255
    np.savez_compressed(path,gray=gray,mask=mask,xy=value['xy'],descriptors=np.empty((0,128),np.float32),
                        background_fraction=value['background_fraction'],metadata_json=np.asarray(json.dumps(meta)))
    with pytest.raises(ValueError):read_view(path,dict(node='moving_scene_1__1'),config())


def test_entire_roster_is_unique_balanced_at_prefix_and_has_no_old_old_pairs():
    moving=[f'moving_{i}' for i in range(120)];protected=[f'drone_{i}' for i in range(112)]
    rows=pair_roster(moving,protected,20261103)
    assert len(rows)==20580
    assert len({(r['left'],r['right']) for r in rows})==20580
    assert sum(r['kind']=='within_moving' for r in rows)==7140
    assert sum(r['kind']=='cross_source' for r in rows[:50])==25
    assert rows==pair_roster(moving,protected,20261103)


def test_ambiguity_and_prior_group_links_cannot_be_lost_across_sources():
    nodes=['moving_a','moving_b','drone_1','drone_2']
    pairs=[dict(left='moving_a',right='drone_1',status='ambiguous')]
    value=conservative_exclusions(nodes,pairs,[('moving_a','moving_b'),('drone_1','drone_2')])
    assert value['moving_nodes_linked_to_protected']==['moving_a','moving_b']
    assert len(value['conservative_exclusion_components'])==1
    assert not value['negative_matching_certifies_independence']
    assert not value['scientific_roles_assigned']
