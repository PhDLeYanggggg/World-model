import io
import zipfile

import numpy as np
import pytest

from src.data_unification.m3w_movingdronecrowd_intake import (
    PublicRangeZip,clip_key,endpoint_support,member_path,mot_record,native_inputs,native_labels,
)
from src.evaluation.m3w_dronecrowd_windows import Recording,WindowSpec,compare_inputs

URL = 'https://huggingface.co/datasets/fyw1999/MovingDroneCrowd/resolve/'+'a'*40+'/MovingDroneCrowd%2B%2B.zip'


class Reply:
    def __init__(self,data,status=206,reported=None):
        self.status_code,self.url,self.raw = status,URL,io.BytesIO(data)
        self.headers = {'Content-Range':reported or f'bytes 0-{len(data)-1}/{len(data)}'}
    def __enter__(self): return self
    def __exit__(self,*_): pass
    def raise_for_status(self): pass


class Session:
    def __init__(self,reply): self.reply=reply
    def get(self,*args,**kw): return self.reply
    def close(self): pass


def test_range_reader_refuses_full_download_and_changed_range():
    for reply in (Reply(b'abcd',200),Reply(b'abcd',reported='bytes 1-4/5'),Reply(b'abc',reported='bytes 0-3/4')):
        with PublicRangeZip(URL,4,maximum_bytes=10,session=Session(reply)) as stream:
            with pytest.raises(ValueError): stream.read(4)


def test_range_reader_is_seekable_and_budgeted():
    with PublicRangeZip(URL,4,maximum_bytes=4,block_size=4,session=Session(Reply(b'abcd'))) as stream:
        assert stream.read(2)==b'ab'
        assert stream.seek(-1,2)==3
        assert stream.read(1)==b'd'
        assert stream.range_requests==1 and stream.transferred_bytes==4
        with pytest.raises(ValueError): stream.fetch(0,0)


def test_zipfile_can_read_and_crc_check_a_remote_member_through_ranges():
    file=io.BytesIO()
    with zipfile.ZipFile(file,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('annotations/scene_1/1.csv','released rows')
    blob=file.getvalue()
    class DynamicSession:
        def get(self,url,headers,**kw):
            start,end=map(int,headers['Range'][6:].split('-'))
            return Reply(blob[start:end+1],reported=f'bytes {start}-{end}/{len(blob)}')
        def close(self): pass
    with PublicRangeZip(URL,len(blob),maximum_bytes=10*len(blob),block_size=16,session=DynamicSession()) as stream:
        with zipfile.ZipFile(stream) as archive:
            assert archive.read('annotations/scene_1/1.csv')==b'released rows'


def test_archive_members_have_namespaced_clip_keys():
    assert clip_key('MovingDroneCrowd++/annotations/scene_1/2.csv','annotations')=='scene_1/2'
    assert clip_key('MovingDroneCrowd++/frames/scene_1/2/20.jpg','frames')=='scene_1/2'
    for name in ('../x.csv','/tmp/x.csv','a\\x.csv','a//x.csv'):
        with pytest.raises(ValueError): member_path(name)


def sample():
    raw='\n'.join(f'{f},1,{f},0,2,2,-1,-1,-1,-1' for f in range(30))
    raw+='\n'+'\n'.join(f'{f},2,0,0,2,2,-1,-1,-1,-1' for f in range(8))
    return mot_record(raw.encode(),'scene_1/1',range(1,31))


def test_mot_geometry_head_center_and_invalid_duplicates():
    record=sample()
    assert record.positions[0,2].tolist()==[3,1]
    assert record.valid.sum()==38
    for raw in (b'0,1,0,0,2,2,-1,-1,-1,-1\n'*2,b'0,1,0,0,-2,2,-1,-1,-1,-1'):
        with pytest.raises(ValueError): mot_record(raw,'x',range(1,31))


def test_missing_image_indices_and_memory_overflow_are_not_silently_repaired():
    raw=b'0,1,0,0,2,2,-1,-1,-1,-1'
    with pytest.raises(ValueError):mot_record(raw,'x',[1,3])
    with pytest.raises(ValueError):mot_record(raw,'x',range(1,31),maximum_cells=1)


def test_public_intake_cannot_follow_an_unpinned_repository_url():
    for url in (URL.replace('https:','http:'),URL.replace('a'*40,'main'),URL.replace('fyw1999','unreviewed')):
        with pytest.raises(ValueError):PublicRangeZip(url,4,maximum_bytes=4)


def test_future_changes_do_not_change_agent_inventory_or_history_inputs():
    record=sample();spec=WindowSpec('native8_12',8,1,tuple(range(1,13)))
    before=native_inputs(record,7,spec)
    assert before['agent_ids'].tolist()==[1,2]
    labels=native_labels(record,before,spec)
    assert labels['complete_target'].tolist()==[True,False]
    positions,valid=record.positions.copy(),record.valid.copy()
    positions[:,8:]+=12345;valid[:,8:]=False
    changed=Recording(record.sequence_id,'not_an_input',record.agent_ids,positions,valid,record.in_bounds)
    assert compare_inputs(before,native_inputs(changed,7,spec))
    assert before['velocity_unit']=='image_pixel_per_native_annotation_step'
    assert 'future_mask' not in before and 'in_bounds_mask' not in before


def test_endpoint_support_keeps_gap_definition_explicit():
    record=sample()
    assert endpoint_support(record,8,12)==dict(endpoint_label_windows=11,complete_interval_windows=11)
    mask=record.valid.copy();mask[0,10]=False
    changed=Recording(record.sequence_id,'x',record.agent_ids,record.positions,mask,record.in_bounds)
    counts=endpoint_support(changed,8,12)
    assert counts['endpoint_label_windows']>counts['complete_interval_windows']
