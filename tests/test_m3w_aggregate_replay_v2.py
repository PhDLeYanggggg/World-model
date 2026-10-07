import io
import json
import subprocess
import sys
import zipfile

import pytest

from scripts import build_m3w_aggregate_replay_v2 as replay


def unpack(path):
    items=replay.members()
    for name,raw in items.items():
        p=path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    return items


def run(path,check=True):
    return subprocess.run([sys.executable,'-I',str(path/'reproduce.py')]+
        (['--check'] if check else []),cwd=path,capture_output=True,text=True,timeout=20)


def test_isolated_replay_and_export_are_exact(tmp_path):
    items=unpack(tmp_path)
    for check in (True,False):
        out=run(tmp_path,check)
        assert out.returncode==0,out.stderr
        report=json.loads(out.stdout)
        assert (report['sources'],report['tables'],report['temporal_contrast_metric_rows'])==(14,6,18)
        assert not report['new_training'] and not report['new_bootstrap']
        assert not report['training_receipts_reverified'] and not report['submission_ready']
    for name in replay.OUTPUTS:
        assert (tmp_path/'replayed'/name).read_bytes()==items['expected/'+name]
        assert items['expected/'+name]==(replay.ROOT/replay.manuscript.OUTPUT/name).read_bytes()


def test_negative_evidence_and_undefined_support_are_preserved():
    evidence=json.loads(replay.members()['expected/evidence.json'])
    assert not evidence['temporal_advance'] and not evidence['risk_calibrated']
    rows={r['comparator']:r for r in evidence['temporal_comparisons']}
    assert rows['rowmean']['all_MSE']['ci_high']<0
    assert rows['rowmean']['full_paired_lower_percent']['ci_high']<0
    assert rows['none']['full_paired_lower_percent']['ci_high']<0
    assert all(r['original_selected_MSE'] is None for r in rows.values())
    assert evidence['temporal_safety']['temporal']['easy_upper_violations']==72
    assert evidence['temporal_safety']['temporal']['finite_completion_supported']==0
    assert [r['violations'] for r in evidence['train_diagnostic']]==[58,57,52]


@pytest.mark.parametrize('member',['inputs/01.json','inputs/09.json','expected/tables.md','reproduce.py'])
def test_tampered_member_is_rejected(tmp_path,member):
    unpack(tmp_path);path=tmp_path/member;path.write_bytes(path.read_bytes()+b'\n')
    out=run(tmp_path)
    assert out.returncode!=0 and 'Archive member changed' in out.stderr


def test_deterministic_small_allowlisted_archive():
    items=replay.members();raw=replay.archive(items)
    assert raw==replay.archive(items) and len(raw)<1024*1024
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        assert set(zipped.namelist())==replay.ALLOWED
        assert all(zipped.read(n)==items[n] for n in zipped.namelist())
    replay.screen(items)


@pytest.mark.parametrize('member',['../outside','/absolute','checkpoint.pt','image.png'])
def test_nonaggregate_members_are_rejected(member):
    items=replay.members();items[member]=b'forbidden'
    with pytest.raises(ValueError,match='allowlist'):
        replay.archive(items)


def test_manifest_cannot_drop_an_expected_member(tmp_path):
    unpack(tmp_path);path=tmp_path/'MANIFEST.json'
    manifest=json.loads(path.read_text());manifest['members'].pop('expected/tables.md')
    path.write_text(json.dumps(manifest))
    out=run(tmp_path)
    assert out.returncode!=0 and 'Manifest allowlist changed' in out.stderr
