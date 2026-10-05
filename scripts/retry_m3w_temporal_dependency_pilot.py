"""One registered retry after the exact pre-training pandas import failure."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts import manage_m3w_temporal_auxiliary_create as manager

PUBLIC = manager.PUBLIC
REGISTRATION = PUBLIC/'dependency_repair_registration.json'
SCRIPT = 'scripts/repair_m3w_temporal_dependencies.py'
FAILED_JOB = '37795593'


def registration():
    paths = [SCRIPT, 'scripts/retry_m3w_temporal_dependency_pilot.py',
             'tests/test_m3w_temporal_dependencies.py', str((PUBLIC/'dependency_repair_protocol.md').relative_to(manager.ROOT))]
    return dict(bindings={p: manager.sha(manager.ROOT/p) for p in paths},
        prior_port_registration_sha256=manager.sha(manager.PORT_REGISTRATION),
        pilot_manifest_sha256=manager.sha(manager.PRIVATE/'create_pilot_input_manifest.json'),
        train_manifest_sha256=manager.sha(manager.PRIVATE/'create_train_input_manifest.json'),
        failed_job_id=FAILED_JOB, numerical_code_unchanged=True, independent_roles_read=False,
        new_package_versions={'pandas': '3.0.3', 'python-dateutil': '2.9.0.post0', 'six': '1.17.0'},
        existing_packages_must_remain_unchanged=True, validation_read=False, deployment_changed=False)


REMOTE = r'''
import hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);assert root==pathlib.Path('/users/k24101830/m3w/european_temporal_auxiliary_v1')
root=root.resolve();a=json.loads(sys.stdin.read());reg=a['registration'];job=reg['failed_job_id'];assert job=='37795593'
assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
intent=root/'dependency_pilot_retry_intent.json'
assert not intent.exists(),'Existing or uncertain dependency retry: inspect, never duplicate'
q=subprocess.run(['sacct','-j',job,'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and q.stdout.strip()=='FAILED|1:0'
q=subprocess.run(['squeue','--me','-h','-o','%j'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and not any(s.startswith('m3w_') for s in q.stdout.splitlines())
assert json.loads((root/'pilot_submission.json').read_text())['job_id']==job
assert "ModuleNotFoundError: No module named 'pandas'" in (root/('pilot-'+job+'.err')).read_text()
assert not (root/('pilot-'+job+'.out')).read_bytes()
assert not (root/'train_submission.json').exists() and not (root/'train_submission_intent.json').exists()
private=root/'data/stage_cvpr2027_experiments/european_temporal_auxiliary_v1'
public=root/'outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1'
assert not list(private.rglob('*.pt.gz')) and not (private/'heartbeat.json').exists()
assert not (public/'pilot.json').exists() and not (public/'training_freeze.json').exists()
for phase in ('pilot','train'):
 p=root/(phase+'_input_manifest.json');assert hashlib.sha256(p.read_bytes()).hexdigest()==reg[phase+'_manifest_sha256']
 for rel,h in json.loads(p.read_text())['code_bindings'].items():
  c=(root/'code'/rel).resolve();assert c.is_relative_to(root/'code') and hashlib.sha256(c.read_bytes()).hexdigest()==h
home=pathlib.Path('/users/k24101830');free=int(os.getxattr(home,'ceph.quota.max_bytes'))-int(os.getxattr(home,'ceph.dir.rbytes'))
assert free>=10*2**30+256*2**20+2*2**20+512*2**20
script=root/'pilot.sbatch';old=script.read_text()
assert old.startswith('#!/bin/bash -l\n') and old.count('\nexec ')==1
assert '--time=02:00:00' in old and '--mem=16G' in old and '--cpus-per-task=4' in old
assert 'scripts.train_m3w_temporal_auxiliary_portable pilot --root /users/k24101830/m3w/european_temporal_auxiliary_v1 --resume' in old
relative='scripts/repair_m3w_temporal_dependencies.py';p=root/'code'/relative
raw=a['dependency_script'].encode();assert hashlib.sha256(raw).hexdigest()==reg['bindings'][relative]
assert not p.exists()
new=old.replace('\nexec ', '\n/users/k24101830/m3w/easy_hurdle_runtime_v2/venv/bin/python scripts/repair_m3w_temporal_dependencies.py --experiment /users/k24101830/m3w/european_temporal_auxiliary_v1 --runtime /users/k24101830/m3w/easy_hurdle_runtime_v2\nexec ',1)
archive=root/'execution_v5_dependency_failure';assert not archive.exists()
with intent.open('x') as f:json.dump(dict(registration_sha256=a['registration_sha256'],failed_job_id=job,
 script_sha256=hashlib.sha256(new.encode()).hexdigest(),quota_free_before=free),f)
archive.mkdir()
hashes={}
for name in ('pilot_submission.json','pilot_submission_intent.json','pilot.sbatch'):
 f=root/name;hashes[name]=hashlib.sha256(f.read_bytes()).hexdigest();os.replace(f,archive/name)
p.write_bytes(raw);script.write_text(new)
(root/'dependency_repair_registration.json').write_text(json.dumps(reg,indent=2)+'\n')
with (root/'pilot_submission_intent.json').open('x') as f:
 json.dump(dict(phase='pilot',manifest_sha256=reg['pilot_manifest_sha256'],dependency_registration_sha256=a['registration_sha256']),f)
try:
 q=subprocess.run(['sbatch','--parsable',str(script)],capture_output=True,text=True,timeout=30)
 result=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
 newjob=q.stdout.strip().split(';')[0]
 if q.returncode==0 and newjob.isdigit():result['job_id']=newjob
except subprocess.TimeoutExpired:result=dict(outcome='unknown_timeout_do_not_resubmit')
(root/'pilot_submission.json').write_text(json.dumps(result,indent=2)+'\n')
out=dict(failed_job_id=job,failed_state='FAILED|1:0',failure='pandas_import_before_training',
 registration_sha256=a['registration_sha256'],prior_hashes=hashes,submission=result,
 new_script_sha256=hashlib.sha256(new.encode()).hexdigest(),scientific_code_changed=False,
 numerical_training_result='not_run_at_submission',validation_read=False,independent_roles_read=False)
(root/'dependency_pilot_retry_receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=['register', 'submit'])
    args = parser.parse_args(); reg = registration()
    if args.phase == 'register': manager.once(REGISTRATION, reg); return
    assert json.loads(REGISTRATION.read_text()) == reg
    manager.verify_registration()
    for p in [*reg['bindings'], str(REGISTRATION.relative_to(manager.ROOT))]:
        assert subprocess.check_output(['git', 'show', 'HEAD:'+p], cwd=manager.ROOT) == (manager.ROOT/p).read_bytes()
    value = dict(registration=reg, registration_sha256=manager.sha(REGISTRATION), dependency_script=(manager.ROOT/SCRIPT).read_text())
    result = manager.remote(REMOTE, [manager.REMOTE], json.dumps(value).encode(), timeout=100)
    manager.once(PUBLIC/'dependency_pilot_retry_receipt.json', result)
    old = manager.PRIVATE/'create_pilot_submission.json'
    assert json.loads(old.read_text())['job_id'] == FAILED_JOB
    archive = manager.PRIVATE/('create_pilot_submission_failed'+FAILED_JOB+'.json')
    assert not archive.exists(); old.rename(archive)
    manager.once(old, result['submission'])
    print(json.dumps(result))


if __name__ == '__main__': main()
