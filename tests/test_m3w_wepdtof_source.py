import numpy as np
import pytest

from scripts.freeze_m3w_wepdtof_blind_camera_plan import windows
from scripts.run_m3w_wepdtof_source_probe import associate, measures
from src.data_unification import m3w_wepdtof_source as m


def test_ambiguous_person_tick_has_no_reference():
    frame = 'call_center_000001'
    files = {1: f'WEPDTOF/frames/call_center/{frame}.jpg'}
    row = dict(image_id=frame, category_id=1, person_id=1)
    doc = dict(images=[dict(id=frame, file_name=frame+'.jpg')], categories=[dict(id=1,name='person')],
               annotations=[row,row,dict(row,person_id=2)])
    people, quality = m.annotation_table(doc,'call_center',files)
    assert people == {2:{1:True}}
    assert quality == dict(raw_rows=3,raw_track_ids=2,ambiguous_identity_ticks=1,excluded_rows=2,
                           annotation_image_entries=1,unlisted_images=0)


def test_disjoint_support_does_not_require_future_labels():
    frames = range(40)
    people = {1:range(40),2:range(8)}
    assert m.support(people,frames,1) == dict(queries=1,targets=2,complete=1,partial=0,absent=1,
                                             queries_with_two_complete=0)
    assert m.pilot_starts(range(100),1,3) == [0,40,80]
    assert m.source('large_office_2') == 'large_office'
    assert m.source('jewelry_store_2') == 'jewelry_store'


def test_unlisted_annotation_image_stays_missing_and_real_gap_breaks_grid():
    frame = 'clip_000001'
    files = {i: f'WEPDTOF/frames/clip/clip_{i:06}.jpg' for i in (1,2)}
    doc = dict(images=[dict(id=frame, file_name=frame+'.jpg')], categories=[dict(id=1,name='person')],
               annotations=[dict(image_id=frame,category_id=1,person_id=1)])
    people, q = m.annotation_table(doc,'clip',files)
    assert people == {1:{1:True}}
    assert q['unlisted_images'] == 1
    frames = list(range(10))+list(range(11,21))
    assert m.support({1:frames,2:frames},frames,1)['queries'] == 0
    assert list(windows(dict(enumerate(range(40))),1)) == [(0,list(range(8))),(20,list(range(20,28)))]
    assert list(windows({t:t for t in frames},1)) == []


def test_fisheye_label_match_is_one_to_one_at_current_frame():
    rows = np.array([[7,0,0,10,10,.8,1],[9,30,30,40,40,.9,1]])
    labels = {12:np.array([5,5,10,10,30]),13:np.array([34,34,10,10,45]),14:np.array([4,4,10,10,0])}
    assert associate(rows,labels) == {(7,12),(9,13)}


def test_missing_position_debit_is_not_dropped():
    v = np.zeros((1,12), dtype=bool)
    v[0,0] = True
    y = np.zeros((1,12,2))
    p = np.zeros((1,2,12,2))
    p[0,0,:,0] = 1.
    out = measures([(y,v,p)])
    assert out['counts'] == dict(matched=1,complete=0,partial=1,absent=0,endpoint=0)
    assert out['observed_and_lower_ols4_vs_cv_pp'] == pytest.approx([100., -1000.])
    assert out['available_fde_cv_ols4_px'] is None
