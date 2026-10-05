"""Register and submit execution shards without changing the scientific experiment."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

from scripts import manage_m3w_temporal_auxiliary_create as base
from src.world_model import m3w_temporal_sharding as execution

REGISTRATION = base.PUBLIC/'parallel_execution_registration.json'
PROTOCOL = base.PUBLIC/'parallel_execution_protocol.md'
EXECUTION_FILES = ('scripts/train_m3w_temporal_shards.py',
                   'src/world_model/m3w_temporal_sharding.py')


def register():
    manifest_path = base.PRIVATE/'create_train_input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cfg = json.loads(base.CONFIG.read_text())
    pilot = json.loads((base.PUBLIC/'create_pilot_verified.json').read_text())
    if pilot['accounting'] != 'COMPLETED|0:0' or pilot['job_id'] != '37798513':
        raise ValueError('Verified real TRAIN pilot required')
    bound = [*EXECUTION_FILES, 'scripts/manage_m3w_temporal_parallel_create.py',
             'tests/test_m3w_temporal_sharding.py', str(PROTOCOL.relative_to(base.ROOT))]
    reg = dict(experiment=base.NAME, execution_revision='four_disjoint_shards_v1',
        original_registration_sha256=base.sha(base.PUBLIC/'registration.json'),
        previous_port_registration_sha256=base.sha(base.PORT_REGISTRATION),
        config_sha256=base.sha(base.CONFIG), manifest_sha256=base.sha(manifest_path),
        pilot_job_id=pilot['job_id'], pilot_sha256=pilot['pilot_sha256'],
        admission=execution.admission(pilot['pilot'], cfg),
        assignment=[sorted(execution.fit_keys(p)) for p in execution.partition(manifest)],
        execution_bindings={p: base.sha(base.ROOT/p) for p in EXECUTION_FILES},
        bindings={p: base.sha(base.ROOT/p) for p in bound},
        jobs=dict(shards=4, max_concurrent=4, cpus_per_job=4, memory_GiB_per_job=16,
                  wall_seconds_per_job=43200, join_cpus=1, join_memory_GiB=2, join_wall_seconds=1200),
        optimizer_unchanged=True, all_216_fits_required_before_readout=True,
        resume_original_pilot=True, global_checkpoint_write_serialization=True,
        checkpoint_cap_bytes=cfg['checkpoint_cap_bytes'], disk_reserve_bytes=cfg['disk_reserve_bytes'],
        atomic_checkpoint_headroom_bytes=cfg['atomic_checkpoint_headroom_bytes'],
        roles_unchanged=True, validation_selection=False, independent_roles_read=False,
        stage5c_executed=False, smc_enabled=False)
    base.once(REGISTRATION, reg)
    return reg


def verified_registration():
    reg = json.loads(REGISTRATION.read_text())
    if (base.sha(base.CONFIG) != reg['config_sha256']
            or base.sha(base.PRIVATE/'create_train_input_manifest.json') != reg['manifest_sha256']
            or base.sha(base.PUBLIC/'registration.json') != reg['original_registration_sha256']
            or base.sha(base.PORT_REGISTRATION) != reg['previous_port_registration_sha256']):
        raise ValueError('Original experiment lineage changed')
    for rel, digest in reg['bindings'].items():
        if base.sha(base.ROOT/rel) != digest: raise ValueError('Execution amendment changed')
    for rel in [*reg['bindings'], str(REGISTRATION.relative_to(base.ROOT))]:
        p = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=base.ROOT, capture_output=True)
        if p.returncode or p.stdout != (base.ROOT/rel).read_bytes():
            raise ValueError('Commit execution amendment before transport')
    return reg


INSTALL = r'''
import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]);reg=json.loads(sys.argv[2])
assert root.parent==pathlib.Path('/users/k24101830/m3w') and root.name==reg['experiment']
assert json.loads((root/'.owner.json').read_text())['registration_sha256']==reg['original_registration_sha256']
assert hashlib.sha256((root/'train_input_manifest.json').read_bytes()).hexdigest()==reg['manifest_sha256']
assert hashlib.sha256((root/'config.json').read_bytes()).hexdigest()==reg['config_sha256']
assert not (root/'train_submission_intent.json').exists() and not (root/'train_submission.json').exists()
checks={}
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar.getmembers():
  assert m.isfile() and m.name in reg['execution_bindings'] and m.name not in checks
  raw=tar.extractfile(m).read();digest=hashlib.sha256(raw).hexdigest();assert digest==reg['execution_bindings'][m.name]
  p=(root/'code'/m.name);p.parent.mkdir(parents=True,exist_ok=True)
  assert p.resolve().is_relative_to((root/'code').resolve())
  if p.exists():assert p.read_bytes()==raw
  else:
   with p.open('xb') as f:f.write(raw)
  checks[m.name]=digest
assert checks==reg['execution_bindings']
raw=sys.argv[2];p=root/'parallel_execution_registration.json'
if p.exists():assert p.read_text()==raw
else:
 with p.open('x') as f:f.write(raw)
print(json.dumps({'installed':checks,'registration_sha256':hashlib.sha256(raw.encode()).hexdigest()}))
'''


SUBMIT = r'''
import hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);expected=sys.argv[2];rt=sys.argv[3]
sys.path.insert(0,str(root/'code'))
from scripts.train_m3w_temporal_shards import verify,once
from src.world_model.m3w_temporal_sharding import PUBLIC
root,cfg,manifest,reg,digest=verify(root);assert digest==expected
for name in ('parallel_submission_intent.json','parallel_submission.json','train_submission_intent.json','train_submission.json'):
 assert not (root/name).exists(),'Submission intent exists: inspect, never duplicate'
q=subprocess.run(['squeue','--me','-h','-o','%j'],capture_output=True,text=True,timeout=20);assert q.returncode==0
assert not any(s.startswith('m3w_temporal_') for s in q.stdout.splitlines()),'Active M3W temporal job'
pilot=json.loads((root/'pilot_submission.json').read_text());assert pilot['job_id']==reg['pilot_job_id']
q=subprocess.run(['sacct','-j',pilot['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and q.stdout.strip()=='COMPLETED|0:0'
home=pathlib.Path('/users/k24101830');free=int(os.getxattr(home,'ceph.quota.max_bytes'))-int(os.getxattr(home,'ceph.dir.rbytes'))
assert free>=cfg['disk_reserve_bytes']+cfg['checkpoint_cap_bytes']+cfg['atomic_checkpoint_headroom_bytes']
assert not (root/PUBLIC/'training_freeze.json').exists()
def script(join=False):
 tag='join' if join else 'shards';cpu=1 if join else 4;mem='2G' if join else '16G';wall='00:20:00' if join else '12:00:00'
 logs='%j' if join else '%A_%a'
 cmd=f'exec {rt}/venv/bin/python -m scripts.train_m3w_temporal_shards '
 cmd+=f'join --root {root}' if join else f'train --root {root} --shard "$SLURM_ARRAY_TASK_ID" --resume'
 return '\n'.join(['#!/bin/bash -l',f'#SBATCH --job-name=m3w_temporal_{tag}', '#SBATCH --account=kcl',
 '#SBATCH --partition=cpu','#SBATCH --qos=normal','#SBATCH --ntasks=1',f'#SBATCH --cpus-per-task={cpu}',
 f'#SBATCH --mem={mem}',f'#SBATCH --time={wall}',f'#SBATCH --output={root}/{tag}-{logs}.out',
 f'#SBATCH --error={root}/{tag}-{logs}.err','#SBATCH --signal=B:TERM@120','set -euo pipefail',
 'module load python/3.11.6-gcc-13.2.0','unset PYTHONPATH',
 f'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS={cpu} OPENBLAS_NUM_THREADS={cpu} MKL_NUM_THREADS={cpu} NUMEXPR_NUM_THREADS={cpu}',
 f'cd {root}/code',cmd,''])
def dispatch(tag,args,extra):
 path=root/(tag+'.sbatch')
 with path.open('x') as f:f.write(script(tag=='join'))
 intent=root/('train_submission_intent.json' if tag=='join' else 'parallel_submission_intent.json')
 once(intent,dict(execution_registration_sha256=digest,manifest_sha256=reg['manifest_sha256'],quota_free=free,**extra))
 try:
  p=subprocess.run(['sbatch','--parsable','--hold',*args,str(path)],capture_output=True,text=True,timeout=30)
  out=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,execution_registration_sha256=digest,**extra)
  job=p.stdout.strip().split(';')[0]
  if p.returncode==0 and job.isdigit():out['job_id']=job
 except subprocess.TimeoutExpired:out=dict(outcome='unknown_timeout_do_not_resubmit',**extra)
 once(root/('train_submission.json' if tag=='join' else 'parallel_submission.json'),out)
 return out
array=dispatch('shards',['--array=0-3%4'],dict(kind='four_disjoint_TRAIN_shards'))
out=dict(array=array)
if 'job_id' in array:
 joined=dispatch('join',['--dependency=afterok:'+array['job_id']],dict(kind='verified_complete_array_join',array_job_id=array['job_id']))
 out['join']=joined
 if 'job_id' in joined:
  releases=[]
  for job in (array['job_id'],joined['job_id']):
   try:
    p=subprocess.run(['scontrol','release',job],capture_output=True,text=True,timeout=20)
    releases.append(dict(job_id=job,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
   except subprocess.TimeoutExpired:releases.append(dict(job_id=job,outcome='unknown_timeout_inspect_before_retry'))
  once(root/'parallel_release.json',releases);out['release']=releases
print(json.dumps(out))
'''


STATUS = r'''
import json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
out={}
for key,name in [('array','parallel_submission.json'),('join','train_submission.json')]:
 p=root/name
 if not p.exists():continue
 receipt=json.loads(p.read_text());row={'submission':receipt};job=receipt.get('job_id')
 if job:
  for title,cmd in [('queue',['squeue','-j',job,'-h','-o','%i|%T|%M|%R']),
   ('accounting',['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,Elapsed,ExitCode,MaxRSS'])]:
   p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
   row[title]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
  if key=='array':
   row['logs']={}
   for i in range(4):
    for suffix in ('out','err'):
     p=root/(f'shards-{job}_{i}.'+suffix);row['logs'][str(i)+'.'+suffix]=p.read_text()[-2000:] if p.exists() else None
  else:
   for suffix in ('out','err'):
    p=root/(f'join-{job}.'+suffix);row[suffix]=p.read_text()[-2000:] if p.exists() else None
 out[key]=row
private=root/'data/stage_cvpr2027_experiments'/root.name;public=root/'outputs/publication_readiness_2026_09'/root.name
out['heartbeats']={str(i):json.loads(p.read_text()) for i in range(4) if (p:=private/'shards'/str(i)/'heartbeat.json').exists()}
out['final_fit_receipts']=len(list((public/'fits').glob('*.json')))
out['checkpoint_count']=len(list(private.rglob('*.pt.gz')))
p=public/'training_freeze.json';out['training_freeze']=json.loads(p.read_text()) if p.exists() else None
out['validation_read']=False;out['remote_modified']=False
print(json.dumps(out))
'''


def submit():
    reg = verified_registration()
    base.inspect()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        for rel in reg['execution_bindings']:
            raw = (base.ROOT/rel).read_bytes(); entry = tarfile.TarInfo(rel); entry.size = len(raw)
            tar.addfile(entry, io.BytesIO(raw))
    installed = base.remote(INSTALL, [base.REMOTE, REGISTRATION.read_text()], buffer.getvalue())
    if installed['registration_sha256'] != base.sha(REGISTRATION): raise ValueError('Transport mismatch')
    out = base.remote(SUBMIT, [base.REMOTE, base.sha(REGISTRATION), base.RUNTIME], timeout=140)
    base.once(base.PRIVATE/'create_parallel_submission.json', out)
    if out.get('join', {}).get('job_id'):
        base.once(base.PRIVATE/'create_train_submission.json', out['join'])
    base.once(base.PUBLIC/'parallel_submission_receipt.json', out)
    print(json.dumps(out), flush=True)


def status():
    out = base.remote(STATUS, [base.REMOTE], timeout=100)
    receipt = base.record_observation(out, 'create_parallel_status')
    print(json.dumps(dict(receipt=receipt, **out)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'submit', 'status'])
    a = p.parse_args()
    if a.phase == 'register': register(); print('Registered execution-only sharding; no training submitted')
    elif a.phase == 'submit': submit()
    else: status()


if __name__ == '__main__': main()
