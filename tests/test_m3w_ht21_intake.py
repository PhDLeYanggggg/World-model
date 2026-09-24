import io
import zipfile

import numpy as np
import pytest

from scripts.fetch_m3w_ht21_annotations import inventory
from src.evaluation.m3w_ht21_intake import (availability,history,human_observation_mask,
    read_ground_truth,read_metadata,audit_ground_truth)


def rows(n=24):
    return np.array([[f,1,2.*f,3.*f,2.,2.,1,1,1.] for f in range(1,n+1)])


def test_history_is_future_invariant_and_backward_difference():
    data=rows(); modified=data.copy(); modified[modified[:,0]>12,2:6]*=100
    expected=history(data,1,12)
    for alternative in (modified,data[data[:,0]<=12]):
        got=history(alternative,1,12)
        for k in ('frame_id','xy','valid_mask','velocity_causal_fd','velocity_valid_mask'):
            np.testing.assert_array_equal(expected[k],got[k])
    np.testing.assert_array_equal(expected['velocity_causal_fd'][1:],np.tile([2,3],(7,1)))
    assert not expected['velocity_valid_mask'][0]


def test_static_tag_not_an_input_or_exclusion():
    data=rows(); static=data.copy(); static[:,7]=2
    assert human_observation_mask(static).all()
    for k,v in history(data,1,12).items():
        np.testing.assert_array_equal(v,history(static,1,12)[k])
    assert availability(data,24)==availability(static,24)


def test_future_support_does_not_define_eligibility():
    data=rows(20)
    result=availability(data,20,lengths=(8,))['8']
    assert result['past_eligible']==13
    assert result['complete_future12']==1
    assert result['partial_future12']==11
    assert result['no_future12']==1
    assert result['endpoint_available']=={'10':3,'25':0,'50':0,'100':0}


def test_gaps_are_not_interpolated():
    data=rows(20); data=data[data[:,0]!=10]
    h=history(data,1,12)
    assert h['valid_mask'].sum()==7
    assert h['velocity_valid_mask'].sum()==5
    np.testing.assert_array_equal(h['xy'][~h['valid_mask']],[[0.,0.]])
    assert availability(data,20,lengths=(8,))['8']['past_eligible']==5


def test_metadata_and_numeric_parser():
    meta=read_metadata(b'[Sequence]\nname=HT-01\nseqLength=24\nframeRate=25\nimWidth=1920\nimHeight=1080\ncam_motion=True\n')
    assert meta['camera_motion'] is True
    s=io.StringIO();np.savetxt(s,rows(),delimiter=',')
    data=read_ground_truth(s.getvalue().encode(),meta)
    audit=audit_ground_truth(data,meta)
    assert audit['rows']==24 and audit['exact_linear_fraction']==1


@pytest.mark.parametrize('mutation', ['duplicate','bad_width','bad_class','nan','outside_frame'])
def test_reject_bad_rows(mutation):
    data=rows()
    if mutation=='duplicate':data[1]=data[0]
    if mutation=='bad_width':data[0,4]=0
    if mutation=='bad_class':data[0,7]=5
    if mutation=='nan':data[0,2]=np.nan
    if mutation=='outside_frame':data[0,0]=30
    s=io.StringIO();np.savetxt(s,data,delimiter=',')
    with pytest.raises(ValueError):read_ground_truth(s.getvalue().encode(),dict(frames=24))


@pytest.mark.parametrize('member',['../gt.txt','/gt.txt','a\\gt.txt','run.py','image.jpg'])
def test_archive_rejects_unsafe_or_nonlabel_member(tmp_path,member):
    p=tmp_path/'fixture.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr(member,'payload')
    with pytest.raises(ValueError):inventory(p)


def test_all_hidden_or_ignored_rows_have_zero_support():
    data=rows(); data[:,7]=3
    assert availability(data,24,lengths=(8,))['8']['past_eligible']==0
    assert not history(data,1,12)['valid_mask'].any()
