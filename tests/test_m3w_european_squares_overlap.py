import numpy as np
import pytest

from src.evaluation.m3w_european_squares_intake import DTYPE
from src.evaluation.m3w_european_squares_overlap import frame_blocks, duplicate_groups


def rows(frames=range(20), offset=0, shift_id=0, perturb=0., static=False):
    a=np.zeros(2*len(frames),dtype=DTYPE)
    for i,f in enumerate(frames):
        for agent in range(2):
            r=a[2*i+agent]
            r['frame']=f+offset
            r['agent']=agent+shift_id
            x=agent*30+(0 if static else f*2)+perturb
            for key,value in zip(('x_min','y_min','x_max','y_max'),(x,10,x+4,18)):
                r[key]=value
    return a


def test_partial_shifted_clip_and_renumbered_agents_match():
    a,sa=frame_blocks(rows())
    b,sb=frame_blocks(rows(range(6,18),offset=100,shift_id=200))
    groups=duplicate_groups([('a',a),('b',b)],'exact')
    assert len(groups)==5
    for group in groups:
        assert group[1]['start_frame']-group[0]['start_frame']==100


def test_frame_geometry_order_does_not_depend_on_row_order():
    a,_=frame_blocks(rows())
    b,_=frame_blocks(rows()[::-1])
    np.testing.assert_array_equal(a,b)


def test_gap_breaks_blocks_and_static_not_duplicate_evidence():
    a,s=frame_blocks(rows([0,1,2,3,5,6,7,8,9]))
    assert len(a)==0 and s['consecutive_blocks']==0
    b,s=frame_blocks(rows(static=True))
    assert s['consecutive_blocks']==13
    assert not b['exact_valid'].any() and not b['quantized_valid'].any()


def test_quantized_candidate_is_not_exact_match():
    a,_=frame_blocks(rows())
    b,_=frame_blocks(rows(perturb=.05))
    assert not duplicate_groups([('a',a),('b',b)],'exact')
    assert len(duplicate_groups([('a',a),('b',b)],'quantized'))==13


def test_same_recording_repetition_not_cross_recording_duplicate():
    a,_=frame_blocks(rows())
    assert not duplicate_groups([('a',np.concatenate([a,a]))],'exact')


def test_missing_person_changes_signature():
    a,_=frame_blocks(rows())
    b,_=frame_blocks(rows()[::2])
    assert not duplicate_groups([('a',a),('b',b)],'exact')


def test_exact_and_quantized_inputs_not_mutated():
    a=rows();before=a.copy()
    frame_blocks(a)
    np.testing.assert_array_equal(a,before)


def test_unknown_mode_refused():
    a,_=frame_blocks(rows())
    with pytest.raises(ValueError):
        duplicate_groups([('a',a)],'made_up')


def test_quantization_has_its_own_order_not_exact_subpixel_order():
    a=rows();b=a.copy()
    for i in range(20):
        for array, offsets in ((a,(.1,.2)),(b,(.2,.1))):
            for j in range(2):
                array['x_min'][2*i+j]=i+offsets[j]
                array['x_max'][2*i+j]=i+offsets[j]+4
                array['y_min'][2*i+j]=j*20
                array['y_max'][2*i+j]=j*20+8
    first,_=frame_blocks(a);second,_=frame_blocks(b)
    assert len(duplicate_groups([('a',first),('b',second)],'quantized'))==13


def test_future_append_leaves_completed_prefix_blocks_unchanged():
    a,_=frame_blocks(rows(range(10)))
    b,_=frame_blocks(rows(range(20)))
    np.testing.assert_array_equal(a,b[b['frame']<=2])
