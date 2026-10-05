"""Archive one verified pre-training shell failure before an explicit pilot retry."""
import json
import os

from scripts import manage_m3w_temporal_auxiliary_create as manager

FAILED_JOB = '37790290'
RECEIPT = manager.PUBLIC/'create_shell_repair.json'

REMOTE = r'''
import hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);assert root==pathlib.Path('/users/k24101830/m3w/european_temporal_auxiliary_v1')
root=root.resolve();a=json.loads(sys.stdin.read());job=a['failed_job_id'];assert job=='37790290'
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_temporal_auxiliary_v1'
receipt=root/'execution_v4_shell_repair.json';archive=root/'execution_v3_shell_failure'
names=('pilot_submission.json','pilot_submission_intent.json','pilot.sbatch')
if receipt.exists():
 out=json.loads(receipt.read_text());assert out['registration_sha256']==a['registration_sha256']
 assert out['failed_job_id']==job
 for name,h in out['archived_hashes'].items():assert hashlib.sha256((archive/name).read_bytes()).hexdigest()==h
 print(json.dumps(out));sys.exit(0)
assert not (root/'train_submission.json').exists() and not (root/'train_submission_intent.json').exists()
q=subprocess.run(['sacct','-j',job,'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and q.stdout.strip()=='FAILED|127:0','Only the exact verified shell failure can be retried'
q=subprocess.run(['squeue','--me','-h','-o','%j'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and not any(s.startswith('m3w_temporal_') for s in q.stdout.splitlines())
err=(root/('pilot-'+job+'.err')).read_text()
assert 'line 14: module: command not found' in err and not (root/('pilot-'+job+'.out')).read_bytes()
private=root/'data/stage_cvpr2027_experiments/european_temporal_auxiliary_v1'
public=root/'outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1'
assert not list(private.rglob('*.pt.gz')) and not (private/'heartbeat.json').exists()
assert not (public/'pilot.json').exists() and not (public/'training_freeze.json').exists()
raw={}
for name in names:
 p=root/name;old=archive/name
 raw[name]=(p if p.exists() else old).read_bytes()
 if old.exists():assert old.read_bytes()==raw[name]
assert json.loads(raw['pilot_submission.json'])['job_id']==job
assert raw['pilot.sbatch'].startswith(b'#!/bin/bash\n')
assert b'module load python/3.11.6-gcc-13.2.0\n' in raw['pilot.sbatch']
assert b' --resume\n' in raw['pilot.sbatch']
previous=json.loads((root/'execution_v3_repair.json').read_text());assert previous==a['root_repair']
chain=previous['file_hashes']['pilot_input_manifest.json']
assert json.loads(raw['pilot_submission_intent.json'])['manifest_sha256']==chain['before']
assert a['pilot_manifest_sha256']==chain['after']
for name,h in a['input_manifest_hashes'].items():
 assert name in ('pilot_input_manifest.json','train_input_manifest.json')
 manifest=root/name;assert hashlib.sha256(manifest.read_bytes()).hexdigest()==h
 for rel,digest in json.loads(manifest.read_text())['code_bindings'].items():
  p=(root/'code'/rel).resolve();assert p.is_relative_to(root/'code')
  assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
assert set(a['input_manifest_hashes'])=={'pilot_input_manifest.json','train_input_manifest.json'}
archive.mkdir(exist_ok=True)
for name in names:
 p=root/name;old=archive/name
 if p.exists():
  if old.exists():assert p.read_bytes()==old.read_bytes();p.unlink()
  else:os.replace(p,old)
out=dict(failed_job_id=job,failed_state='FAILED|127:0',failure='module_not_found_before_training',
 registration_sha256=a['registration_sha256'],archived_hashes={k:hashlib.sha256(v).hexdigest() for k,v in raw.items()},
 input_manifest_hashes=a['input_manifest_hashes'],scientific_code_verified_unchanged=True,
 optimizer_updates_before_retry=0,checkpoints_before_retry=0,prior_logs_preserved=True,
 new_job_submitted=False,validation_read=False,independent_roles_read=False)
with receipt.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
'''


def main():
    manager.verify_registration()
    manifests = {phase+'_input_manifest.json': manager.sha(manager.PRIVATE/('create_'+phase+'_input_manifest.json'))
                 for phase in ('pilot', 'train')}
    a = dict(failed_job_id=FAILED_JOB, registration_sha256=manager.sha(manager.PORT_REGISTRATION),
             pilot_manifest_sha256=manifests['pilot_input_manifest.json'], input_manifest_hashes=manifests,
             root_repair=json.loads((manager.PUBLIC/'create_root_repair.json').read_text()))
    result = manager.remote(REMOTE, [manager.REMOTE], json.dumps(a).encode())
    assert result['failed_job_id'] == FAILED_JOB and result['input_manifest_hashes'] == manifests
    manager.once(RECEIPT, result)
    path = manager.PRIVATE/'create_pilot_submission.json'
    archived = manager.PRIVATE/'create_pilot_submission_failed37790290.json'
    if path.exists():
        raw = path.read_bytes(); assert json.loads(raw)['job_id'] == FAILED_JOB
        if archived.exists(): assert archived.read_bytes() == raw; path.unlink()
        else: os.replace(path, archived)
    else: assert json.loads(archived.read_text())['job_id'] == FAILED_JOB
    print(json.dumps(result))


if __name__ == '__main__': main()
