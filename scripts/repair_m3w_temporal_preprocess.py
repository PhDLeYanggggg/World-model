"""Apply a measured scalar-portability guard before retrying an untrained pilot."""
import base64
import json
import os
import subprocess
from scripts import manage_m3w_temporal_auxiliary_create as manager

PORT = 'scripts/train_m3w_temporal_auxiliary_portable.py'
GUARD = 'src/world_model/m3w_preprocess_portability.py'


def request():
    manager.verify_registration()
    previous = json.loads((manager.PUBLIC/'create_port_registration_v5.json').read_text())
    old = subprocess.check_output(['git','show','56fc3d71:'+PORT],cwd=manager.ROOT)
    import hashlib
    assert hashlib.sha256(old).hexdigest()==previous['bindings'][PORT]
    encode=lambda b:base64.b64encode(b).decode() if b is not None else None
    files={'code/'+PORT:dict(before=encode(old),after=encode((manager.ROOT/PORT).read_bytes())),
           'code/'+GUARD:dict(before=None,after=encode((manager.ROOT/GUARD).read_bytes()))}
    for phase in ('pilot','train'):
        p=manager.PRIVATE/('create_'+phase+'_input_manifest.json')
        archive=p.with_name('create_'+phase+'_input_manifest_v5.json')
        raw=(archive if archive.exists() else p).read_bytes();manifest=json.loads(raw)
        assert manifest['code_bindings'][PORT]==previous['bindings'][PORT] and GUARD not in manifest['code_bindings']
        manifest['code_bindings'][PORT]=manager.sha(manager.ROOT/PORT)
        manifest['code_bindings'][GUARD]=manager.sha(manager.ROOT/GUARD)
        files[phase+'_input_manifest.json']=dict(before=encode(raw),after=encode((json.dumps(manifest,indent=2)+'\n').encode()))
    return dict(files=files,registration_sha256=manager.sha(manager.PORT_REGISTRATION),
        diagnostic_sha256=manager.sha(manager.PUBLIC/'preprocess_portability_diagnostic.json'))


REMOTE=r'''
import base64,hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);assert root==pathlib.Path('/users/k24101830/m3w/european_temporal_auxiliary_v1')
root=root.resolve();a=json.loads(sys.stdin.read());job='37797054'
assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
assert json.loads((root/'pilot_submission.json').read_text())['job_id']==job
assert not (root/'preprocess_repair_intent.json').exists(),'Existing repair: inspect before any retry'
assert not (root/'train_submission.json').exists() and not (root/'train_submission_intent.json').exists()
for ident,wanted in [(job,'FAILED|1:0'),('37798223','COMPLETED|0:0')]:
 q=subprocess.run(['sacct','-j',ident,'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
 assert q.returncode==0 and q.stdout.strip()==wanted
q=subprocess.run(['squeue','--me','-h','-o','%j'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and not any(s.startswith('m3w_') for s in q.stdout.splitlines())
error=(root/('pilot-'+job+'.err')).read_text();assert 'core.exact(pr, fresh)' in error and 'AssertionError' in error
p=root/'preprocess_portability_diagnostic.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==a['diagnostic_sha256']
d=json.loads(p.read_text());assert d['unique_packets']==24 and d['optimizer_updates']==0 and not d['validation_read']
assert all(f['exact'] or f['field']=='scale' for g in d['groups'] for f in g['fields'])
private=root/'data/stage_cvpr2027_experiments/european_temporal_auxiliary_v1'
public=root/'outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1'
assert not list(private.rglob('*.pt.gz')) and not (public/'pilot.json').exists() and not (public/'training_freeze.json').exists()
hb=json.loads((private/'heartbeat.json').read_text());assert hb['job_id']==job and hb['state']=='portable_phase_started'
events=[json.loads(line) for line in (private/'events.jsonl').read_text().splitlines()]
assert events==[hb]
allowed={'code/scripts/train_m3w_temporal_auxiliary_portable.py','code/src/world_model/m3w_preprocess_portability.py','pilot_input_manifest.json','train_input_manifest.json'}
assert set(a['files'])==allowed
values={rel:{k:base64.b64decode(v,validate=True) if v is not None else None for k,v in pair.items()} for rel,pair in a['files'].items()}
for rel,pair in values.items():
 assert set(pair)=={'before','after'} and pair['after'] is not None
 p=(root/rel).resolve();assert p.is_relative_to(root)
 assert not p.exists() if pair['before'] is None else p.read_bytes()==pair['before']
archive=root/'execution_v5_before_preprocess_fix';assert not archive.exists()
with (root/'preprocess_repair_intent.json').open('x') as f:json.dump(dict(registration_sha256=a['registration_sha256']),f)
archive.mkdir()
hashes={}
for rel,pair in values.items():
 p=root/rel
 if pair['before'] is not None:
  old=archive/rel;old.parent.mkdir(parents=True,exist_ok=True);old.write_bytes(pair['before'])
 temp=p.with_name(p.name+'.repair-tmp');p.parent.mkdir(parents=True,exist_ok=True)
 with temp.open('wb') as f:f.write(pair['after']);f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
 hashes[rel]={k:hashlib.sha256(v).hexdigest() if v is not None else None for k,v in pair.items()}
prior={}
for rel in ('pilot_submission.json','pilot_submission_intent.json','pilot.sbatch',str((private/'heartbeat.json').relative_to(root)),str((private/'events.jsonl').relative_to(root))):
 p=root/rel;old=archive/rel;old.parent.mkdir(parents=True,exist_ok=True)
 prior[rel]=hashlib.sha256(p.read_bytes()).hexdigest();os.replace(p,old)
out=dict(result_source='fresh_run_registered_scale_only_portability_repair',failed_job_id=job,
 registration_sha256=a['registration_sha256'],diagnostic_sha256=a['diagnostic_sha256'],file_hashes=hashes,
 archived_previous_records=prior,scale_comparison_float64_ULPs=4,all_other_fields_exact=True,
 original_TRAIN_preprocessing_authoritative=True,optimizer_code_changed=False,exact_resume_comparator_changed=False,
 optimizer_updates_before_repair=0,new_job_submitted=False,validation_read=False,independent_roles_read=False)
(root/'preprocess_repair_receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''


def main():
    a=request();r=manager.remote(REMOTE,[manager.REMOTE],json.dumps(a).encode(),timeout=100)
    manager.once(manager.PUBLIC/'preprocess_repair_receipt.json',r)
    for phase in ('pilot','train'):
        p=manager.PRIVATE/('create_'+phase+'_input_manifest.json')
        archive=p.with_name('create_'+phase+'_input_manifest_v5.json')
        pair=a['files'][phase+'_input_manifest.json'];assert p.read_bytes()==base64.b64decode(pair['before'])
        assert not archive.exists();os.replace(p,archive);p.write_bytes(base64.b64decode(pair['after']))
    p=manager.PRIVATE/'create_pilot_submission.json'
    assert json.loads(p.read_text())['job_id']=='37797054'
    archive=p.with_name('create_pilot_submission_failed37797054.json');assert not archive.exists();os.replace(p,archive)
    print(json.dumps(r))


if __name__=='__main__':main()
