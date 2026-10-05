import io
import json
import subprocess
import sys
import zipfile

import pytest

from scripts import build_m3w_aggregate_replay_package as replay


def unpack(tmp_path):
    members = replay.members()
    for name,raw in members.items():
        p=tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
    return members


def test_isolated_stdlib_replay_matches_all_saved_outputs(tmp_path):
    members=unpack(tmp_path)
    result=subprocess.run([sys.executable,'-I',str(tmp_path/'reproduce.py'),'--check'],
                          cwd=tmp_path,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    r=json.loads(result.stdout)
    assert r['status']=='aggregate_replay_verified'
    assert (r['pinned_sources'],r['tables'],r['european_contrasts'],r['sdd_contrasts'])==(8,3,11,4)
    assert not r['new_training'] and not r['new_bootstrap'] and not r['independent_confirmation']
    for name in replay.OUTPUTS:
        assert members['expected/'+name]==(replay.original.ROOT/replay.original.OUTPUT/name).read_bytes()


@pytest.mark.parametrize('target',['inputs/01.json','expected/tables.md','reproduce.py'])
def test_modified_member_is_rejected_before_replay(tmp_path,target):
    unpack(tmp_path); p=tmp_path/target; p.write_bytes(p.read_bytes()+b'\n')
    r=subprocess.run([sys.executable,'-I',str(tmp_path/'reproduce.py'),'--check'],
                     cwd=tmp_path,capture_output=True,text=True,timeout=20)
    assert r.returncode!=0 and 'Archive member changed' in r.stderr


def test_rebuild_really_writes_equal_tables_and_csv(tmp_path):
    unpack(tmp_path)
    r=subprocess.run([sys.executable,'-I',str(tmp_path/'reproduce.py')],cwd=tmp_path,
                     capture_output=True,text=True,timeout=20)
    assert r.returncode==0,r.stderr
    for name in replay.OUTPUTS:
        assert (tmp_path/'replayed'/name).read_bytes()==(tmp_path/'expected'/name).read_bytes()


def test_zip_is_deterministic_allowlisted_and_small():
    members=replay.members(); raw=replay.archive(members)
    assert raw==replay.archive(members) and len(raw)<1024*1024
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert sorted(z.namelist())==sorted(members)
        for info in z.infolist():
            assert not info.filename.startswith('/') and '..' not in info.filename.split('/')
            assert not info.extra and not info.comment
            assert z.read(info.filename)==members[info.filename]
    e=json.loads(members['expected/evidence.json'])
    assert not e['submission_ready'] and e['european_contrasts'][8]['localities']==11
    assert next(x for x in e['european_policies'] if x['arm']=='extended')['violations']==7


@pytest.mark.parametrize('identifier',[b'/Users/person/data',b'/users/account/project',
                         b'/cephfs/volumes/home',b'PhDLeYanggggg',b'k24101830',
                         b'person@university.ac.uk',b'github.com/owner/repo'])
def test_identifier_scan_is_sensitive(identifier):
    with pytest.raises(ValueError,match='Identifier'):
        replay.screen({'README.md':identifier})


@pytest.mark.parametrize('name',['../outside','/absolute','inputs/model.pt','video.mp4'])
def test_unexpected_members_rejected(name):
    members=replay.members(); members[name]=b'bad'
    with pytest.raises(ValueError,match='allowlist'):replay.archive(members)
