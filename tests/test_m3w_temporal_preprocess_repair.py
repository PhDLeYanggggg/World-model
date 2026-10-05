import base64
import hashlib
import io
import json
from types import SimpleNamespace

import pytest
from scripts import repair_m3w_temporal_preprocess as repair


def fixture(tmp_path, monkeypatch):
    root=tmp_path/'owned';root.mkdir()
    (root/'.owner.json').write_text('{"experiment":"owned"}')
    for name,raw in [('pilot_submission.json','{"job_id":"37797054"}'),('pilot_submission_intent.json','{}'),('pilot.sbatch','original'),('pilot-37797054.err','core.exact(pr, fresh)\nAssertionError')]:
        (root/name).write_text(raw)
    private=root/'data/stage_cvpr2027_experiments/european_temporal_auxiliary_v1';private.mkdir(parents=True)
    hb=dict(job_id='37797054',state='portable_phase_started')
    (private/'heartbeat.json').write_text(json.dumps(hb));(private/'events.jsonl').write_text(json.dumps(hb)+'\n')
    diag=dict(unique_packets=24,optimizer_updates=0,validation_read=False,groups=[dict(fields=[dict(field='scale',exact=False)])])
    p=root/'preprocess_portability_diagnostic.json';p.write_text(json.dumps(diag))
    files={}
    for rel in ('code/'+repair.PORT,'code/'+repair.GUARD,'pilot_input_manifest.json','train_input_manifest.json'):
        before=None if rel=='code/'+repair.GUARD else b'old';after=b'new'
        if before is not None:
            f=root/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(before)
        files[rel]=dict(before=base64.b64encode(before).decode() if before is not None else None,after=base64.b64encode(after).decode())
    payload=dict(files=files,registration_sha256='a'*64,diagnostic_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    state=dict(queue='',pilot='FAILED|1:0',commands=[])
    def command(argv,**kw):
        state['commands'].append(argv);assert argv[0] in ('sacct','squeue')
        out=state['queue'] if argv[0]=='squeue' else state['pilot'] if '37797054' in argv else 'COMPLETED|0:0'
        return SimpleNamespace(returncode=0,stdout=out,stderr='')
    monkeypatch.setattr(repair.subprocess,'run',command)
    def execute():
        monkeypatch.setattr('sys.argv',['',str(root)]);monkeypatch.setattr('sys.stdin',io.StringIO(json.dumps(payload)))
        exec(repair.REMOTE.replace('/users/k24101830/m3w/european_temporal_auxiliary_v1',str(root)),{})
    return root,private,state,payload,execute


def test_repair_archives_prior_work_and_never_submits(tmp_path, monkeypatch, capsys):
    root,private,state,payload,execute=fixture(tmp_path,monkeypatch)
    execute();r=json.loads(capsys.readouterr().out)
    assert r['original_TRAIN_preprocessing_authoritative'] and not r['new_job_submitted']
    assert not (root/'pilot_submission.json').exists() and not (private/'heartbeat.json').exists()
    assert (root/'execution_v5_before_preprocess_fix/pilot_submission.json').exists()
    for rel in payload['files']:assert (root/rel).read_bytes()==b'new'
    with pytest.raises((AssertionError,FileNotFoundError)):execute()
    assert not any(c[0]=='sbatch' for c in state['commands'])


@pytest.mark.parametrize('bad',['running','active','checkpoint','progress','code','diagnostic','full_train'])
def test_existing_training_or_changed_evidence_blocks_before_mutation(tmp_path, monkeypatch, bad):
    root,private,state,payload,execute=fixture(tmp_path,monkeypatch)
    if bad=='running':state['pilot']='RUNNING|0:0'
    elif bad=='active':state['queue']='m3w_temporal_train'
    elif bad=='checkpoint':(private/'checkpoint.pt.gz').write_bytes(b'work')
    elif bad=='progress':(private/'events.jsonl').write_text('{"state":"optimizer_update"}')
    elif bad=='code':(root/'code'/repair.PORT).write_bytes(b'unexpected')
    elif bad=='diagnostic':payload['diagnostic_sha256']='b'*64
    else:(root/'train_submission.json').write_text('{}')
    with pytest.raises(AssertionError):execute()
    assert not (root/'preprocess_repair_intent.json').exists()
    assert (root/'pilot_submission.json').exists()
