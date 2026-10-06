"""Owned, read-only-model TRAIN diagnostic on CREATE; never resubmit implicitly."""
import argparse
import io
import json
import subprocess
import tarfile

from scripts import manage_m3w_temporal_auxiliary_create as base
from scripts import diagnose_m3w_temporal_false_safe as diagnostic

HOME = base.PUBLIC/diagnostic.DIAG
REG = HOME/'registration.json'
FILES = ('scripts/diagnose_m3w_temporal_false_safe.py',
         'src/world_model/m3w_false_safe_diagnostic.py')


def register():
    movement = json.loads((HOME/'movement_verification.json').read_text())
    assert movement['training_freeze_sha256'] == base.sha(base.PUBLIC/'training_freeze.json')
    assert movement['manifest_sha256'] == base.sha(base.PRIVATE/'create_train_input_manifest.json')
    bound = [*FILES, 'scripts/manage_m3w_false_safe_diagnostic.py',
             'tests/test_m3w_false_safe_diagnostic.py', str((HOME/'protocol.md').relative_to(base.ROOT))]
    value = dict(experiment=base.NAME, diagnostic=diagnostic.DIAG,
        training_freeze_sha256=movement['training_freeze_sha256'], manifest_sha256=movement['manifest_sha256'],
        original_registration_sha256=base.sha(base.PUBLIC/'registration.json'), movement=movement,
        bindings={p: base.sha(base.ROOT/p) for p in bound}, execution_files=list(FILES),
        cpus=4, memory_GiB=16, wall_seconds=7200, risk_budget=.02, new_optimizer_updates=0,
        resubstitution_only=True, validation_scored=False, independent_roles_read=False,
        threshold_selection=False, new_checkpoint_written=False, deployment_changed=False)
    base.once(REG, value)
    return value


INSTALL_SUBMIT = r'''
import hashlib,io,json,os,pathlib,subprocess,sys,tarfile
root=pathlib.Path(sys.argv[1]);rt=sys.argv[2];regraw=sys.argv[3];reg=json.loads(regraw);tag=reg['diagnostic']
assert root==pathlib.Path('/users/k24101830/m3w/european_temporal_auxiliary_v1')
assert json.loads((root/'.owner.json').read_text())['registration_sha256']==reg['original_registration_sha256']
assert hashlib.sha256((root/'train_input_manifest.json').read_bytes()).hexdigest()==reg['manifest_sha256']
home=root/'outputs/publication_readiness_2026_09'/root.name
assert hashlib.sha256((home/'training_freeze.json').read_bytes()).hexdigest()==reg['training_freeze_sha256']
assert not (root/(tag+'_submission_intent.json')).exists(),'Inspect previous intent; do not resubmit'
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%P|%T|%C'],capture_output=True,text=True,timeout=20);assert q.returncode==0
assert not any('m3w_false_safe' in s for s in q.stdout.splitlines())
quota=pathlib.Path('/users/k24101830');free=int(os.getxattr(quota,'ceph.quota.max_bytes'))-int(os.getxattr(quota,'ceph.dir.rbytes'))
assert free>=10*2**30+50*2**20,'Retain storage reserve'
seen=set()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for m in t.getmembers():
  assert m.isfile() and m.name in reg['bindings'] and m.name not in seen
  raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==reg['bindings'][m.name]
  p=root/'code'/m.name;assert p.resolve().is_relative_to((root/'code').resolve());p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==raw
  else:
   with p.open('xb') as f:f.write(raw)
  seen.add(m.name)
assert seen==set(reg['bindings'])
with (root/(tag+'_registration.json')).open('x') as f:f.write(regraw)
script='\n'.join(['#!/bin/bash -l','#SBATCH --job-name=m3w_false_safe_train', '#SBATCH --account=kcl',
 '#SBATCH --partition=cpu','#SBATCH --qos=normal','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=4',
 '#SBATCH --mem=16G','#SBATCH --time=02:00:00',f'#SBATCH --output={root}/{tag}-%j.out',
 f'#SBATCH --error={root}/{tag}-%j.err','set -euo pipefail','module load python/3.11.6-gcc-13.2.0',
 'unset PYTHONPATH','export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4',
 f'cd {root}/code',f'exec {rt}/venv/bin/python -m scripts.diagnose_m3w_temporal_false_safe run --root {root}',''])
path=root/(tag+'.sbatch')
with path.open('x') as f:f.write(script)
intent=dict(registration_sha256=hashlib.sha256(regraw.encode()).hexdigest(),quota_free=free,
 m3w_queue=[s for s in q.stdout.splitlines() if '|m3w_' in s],other_jobs_untouched=True)
with (root/(tag+'_submission_intent.json')).open('x') as f:json.dump(intent,f)
try:
 p=subprocess.run(['sbatch','--parsable',str(path)],capture_output=True,text=True,timeout=30)
 out=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,**intent)
 job=p.stdout.strip().split(';')[0]
 if p.returncode==0 and job.isdigit():out['job_id']=job
except subprocess.TimeoutExpired:out=dict(outcome='unknown_timeout_inspect_do_not_resubmit',**intent)
with (root/(tag+'_submission.json')).open('x') as f:json.dump(out,f)
print(json.dumps(out))
'''


def submit():
    reg = register()
    for rel in [*reg['bindings'], str(REG.relative_to(base.ROOT))]:
        q = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=base.ROOT, capture_output=True)
        assert q.returncode == 0 and q.stdout == (base.ROOT/rel).read_bytes(), 'Commit before execution'
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        for rel in reg['bindings']:
            raw = (base.ROOT/rel).read_bytes(); entry = tarfile.TarInfo(rel); entry.size = len(raw)
            t.addfile(entry, io.BytesIO(raw))
    out = base.remote(INSTALL_SUBMIT, [base.REMOTE, base.RUNTIME, REG.read_text()], buf.getvalue(), timeout=90)
    base.once(HOME/'submission.json', out)
    return out


OBSERVE = r'''
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);tag=sys.argv[2];collect=sys.argv[3]=='collect'
assert root==pathlib.Path('/users/k24101830/m3w/european_temporal_auxiliary_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
r=json.loads((root/(tag+'_submission.json')).read_text());job=r['job_id'];assert job.isdigit()
out=dict(submission=r)
for key,cmd in [('queue',['squeue','-j',job,'-h','-o','%i|%T|%M|%R']),
 ('accounting',['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,Elapsed,ExitCode,MaxRSS'])]:
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=20);out[key]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
home=root/'outputs/publication_readiness_2026_09'/root.name/tag
for name in ['heartbeat.json','complete.json']:
 p=home/name;out[name]=json.loads(p.read_text()) if p.exists() else None
for suffix in ['out','err']:
 p=root/(tag+'-'+job+'.'+suffix);out[suffix]=p.read_text()[-2000:] if p.exists() else None
if collect:
 assert out['complete.json'] and job+'|COMPLETED|' in out['accounting']['stdout']
 rows={};total=0
 for ref in out['complete.json']['heads']:
  p=(root/ref['path']).resolve();assert p.is_relative_to(home.resolve()/'heads')
  raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];total+=len(raw);assert total<=8*2**20
  rows[ref['path']]=raw.decode()
 out['files']=rows
print(json.dumps(out))
'''


def observe(mode):
    out = base.remote(OBSERVE, [base.REMOTE, diagnostic.DIAG, mode], timeout=90)
    if mode == 'collect':
        assert out['submission']['registration_sha256'] == base.sha(REG)
        assert out['complete.json']['registration_sha256'] == base.sha(REG)
        assert out['complete.json']['total_heads'] == 216
        for rel, text in out.pop('files').items():
            path = (base.ROOT/rel).resolve(); assert path.is_relative_to(HOME.resolve()/'heads')
            base.once(path, json.loads(text))
        base.once(HOME/'complete.json', out['complete.json'])
        base.once(HOME/'collection.json', out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['register', 'submit', 'status', 'collect']); a = p.parse_args()
    result = register() if a.mode == 'register' else submit() if a.mode == 'submit' else observe(a.mode)
    if a.mode == 'register':
        result = dict(registered=True, sha256=base.sha(REG), groups=len(result['movement']['groups']))
    if result.get('complete.json'):
        result = {**result, 'complete.json': {k:v for k,v in result['complete.json'].items() if k != 'heads'}}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
