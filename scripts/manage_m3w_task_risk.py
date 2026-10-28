"""Register and execute matched TRAIN-only decision objectives in an owned job."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from scripts import manage_m3w_temporal_auxiliary_create as base
from scripts import train_m3w_task_risk as run

PUBLIC = base.ROOT/run.PUBLIC
CONFIG = base.ROOT/'configs'/('m3w_'+run.NAME+'.json')
REG = PUBLIC/'registration.json'
REMOTE = '/users/k24101830/m3w/'+run.NAME


def register():
    from scripts.export_m3w_easy_hurdle_create import closure
    cfg = json.loads(CONFIG.read_text())
    manifest = json.loads((base.PRIVATE/'create_train_input_manifest.json').read_text())
    movement = base.PUBLIC/'false_safe_train_diagnostic_v1/movement_verification.json'
    files = closure(base.ROOT, ['scripts.train_m3w_task_risk', 'scripts.manage_m3w_task_risk'])
    files += [Path(__file__).resolve(), CONFIG, PUBLIC/'protocol.md']
    files += [base.ROOT/'tests'/('test_m3w_task_risk_'+suffix+'.py')
              for suffix in ('objective', 'training', 'execution')]
    if (cfg['head_training'] != json.loads(base.CONFIG.read_text())['head_training']
            or len(manifest['heads']) != 72 or cfg['neural_fits'] != 216
            or cfg['arms'] != ['quadratic', 'task_utility', 'task_risk']):
        raise ValueError('Fixed contemporary matched comparison required')
    value = dict(experiment=run.NAME, parent=run.PARENT, config_sha256=base.sha(CONFIG),
        input_manifest_sha256=base.sha(base.PRIVATE/'create_train_input_manifest.json'),
        parent_training_freeze_sha256=base.sha(base.PUBLIC/'training_freeze.json'),
        parent_registration_sha256=base.sha(base.PUBLIC/'registration.json'),
        movement_sha256=base.sha(movement), movement=json.loads(movement.read_text()),
        previous_failed_readout_sha256=base.sha(base.ROOT/'outputs/publication_readiness_2026_09'
            '/european_easy_harm_deviance_v1/readout_verified_v3/summary.json'),
        bindings={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(set(files))},
        expected=[r['identity'] for r in manifest['heads']],
        full_fits=216, source_identities=72, steps=2000, risk_budget=.02,
        bootstrap_resamples=3000, bootstrap_seed=20261028,
        source_packets_copied=False, checkpoint_selection=False, threshold_search=False,
        independent_roles_read=False, new_forecaster=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)
    base.once(REG, value)
    return value


def controller_fields(record, job, phase):
    if record.get('returncode') != 0:
        return {}
    f = dict(x.split('=', 1) for x in shlex.split(record['stdout']) if '=' in x)
    if (f.get('JobId') != job or f.get('JobName') != 'm3w_task_risk_'+phase
            or f.get('WorkDir') != REMOTE):
        return {}
    return f


def controller_success(record, job, phase):
    f = controller_fields(record, job, phase)
    return f.get('JobState') == 'COMPLETED' and f.get('ExitCode') == '0:0'


INSTALL = r'''
import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]);source=pathlib.Path(sys.argv[2]);regraw=sys.argv[3];cfgraw=sys.argv[4]
reg=json.loads(regraw);assert root==pathlib.Path('/users/k24101830/m3w/european_task_risk_v1')
assert source==pathlib.Path('/users/k24101830/m3w/european_temporal_auxiliary_v1')
assert json.loads((source/'.owner.json').read_text())['registration_sha256']==reg['parent_registration_sha256']
assert hashlib.sha256((source/'train_input_manifest.json').read_bytes()).hexdigest()==reg['input_manifest_sha256']
assert hashlib.sha256(cfgraw.encode()).hexdigest()==reg['config_sha256']
assert not root.is_symlink();root.mkdir(exist_ok=True)
def exact(path,raw):
 assert not path.is_symlink()
 if path.exists():assert path.read_bytes()==raw
 else:
  with path.open('xb') as f:f.write(raw)
owner=dict(experiment=reg['experiment'],registration_sha256=hashlib.sha256(regraw.encode()).hexdigest())
exact(root/'.owner.json',(json.dumps(owner,indent=2)+'\n').encode())
exact(root/'registration.json',regraw.encode());exact(root/'config.json',cfgraw.encode())
seen={}
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar.getmembers():
  assert m.isfile() and m.name in reg['bindings'] and m.name not in seen
  raw=tar.extractfile(m).read();h=hashlib.sha256(raw).hexdigest();assert h==reg['bindings'][m.name]
  p=root/'code'/m.name;assert p.resolve().is_relative_to((root/'code').resolve())
  p.parent.mkdir(parents=True,exist_ok=True);exact(p,raw);seen[m.name]=h
assert seen==reg['bindings']
print(json.dumps(dict(code_bindings=len(seen),source_packets_copied=False,source_modified=False)))
'''


SUBMIT = r'''
import json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);runtime=sys.argv[2];phase=sys.argv[3];expected=sys.argv[4];resume=sys.argv[5]=='resume'
sys.path.insert(0,str(root/'code'))
from scripts.train_m3w_task_risk import verify,quota,sha,once,PUBLIC
from scripts.manage_m3w_task_risk import controller_fields,controller_success
root,source,cfg,reg,manifest=verify(root);assert sha(root/'registration.json')==expected
assert phase in ('pilot','train')
resources=quota(root,cfg)
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%P|%T|%C'],capture_output=True,text=True,timeout=20);assert q.returncode==0
assert not any('|m3w_' in s for s in q.stdout.splitlines()),'Avoid concurrent owned training allocations'
def control(job):
 p=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=20)
 return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
artifact=root/PUBLIC/('pilot.json' if phase=='pilot' else 'training_freeze.json')
assert not artifact.exists(),'Completed phase is immutable'
active=root/(phase+'_submission.json');attempt=1;prior=None
if resume:
 prior=json.loads(active.read_text());record=control(prior['job_id'])
 f=controller_fields(record,prior['job_id'],phase)
 assert f.get('JobState') in ('FAILED','TIMEOUT','CANCELLED','PREEMPTED','OUT_OF_MEMORY'),'Observe real terminal failure before resume'
 attempt=prior['attempt']+1
elif active.exists() or list(root.glob(phase+'_attempt*_intent.json')):
 raise RuntimeError('Existing submission/intent; inspect, never retry blindly')
pilot_controller=None
if phase=='train':
 pilot=json.loads((root/PUBLIC/'pilot.json').read_text());assert pilot['engineering_pass']
 sub=json.loads((root/'pilot_submission.json').read_text());assert sub['job_id']==pilot['job_id']
 pilot_controller=control(sub['job_id']);assert controller_success(pilot_controller,sub['job_id'],'pilot')
wall='02:00:00' if phase=='pilot' else '12:00:00';kind=phase+'_attempt'+str(attempt)
cmd=f'exec {runtime}/venv/bin/python -m scripts.train_m3w_task_risk {phase} --root {root}'
if resume:cmd+=' --resume'
script='\n'.join(['#!/bin/bash -l',f'#SBATCH --job-name=m3w_task_risk_{phase}',
 '#SBATCH --account=kcl','#SBATCH --partition=cpu','#SBATCH --qos=normal','#SBATCH --ntasks=1',
 '#SBATCH --cpus-per-task=4','#SBATCH --mem=8G',f'#SBATCH --time={wall}','#SBATCH --no-requeue',
 f'#SBATCH --chdir={root}',f'#SBATCH --output={root}/{phase}-%j.out',f'#SBATCH --error={root}/{phase}-%j.err',
 '#SBATCH --signal=B:TERM@120','set -euo pipefail','module load python/3.11.6-gcc-13.2.0','unset PYTHONPATH',
 'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4',
 f'cd {root}/code',cmd,''])
once(root/(kind+'_intent.json'),dict(registration_sha256=expected,resources=resources,
 resume=resume,prior=prior,m3w_queue=[],other_jobs_untouched=True,pilot_controller=pilot_controller))
path=root/(kind+'.sbatch')
with path.open('x') as f:f.write(script)
try:
 p=subprocess.run(['sbatch','--parsable','--hold',str(path)],capture_output=True,text=True,timeout=30)
 out=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,registration_sha256=expected,attempt=attempt)
 job=p.stdout.strip().split(';')[0]
 if p.returncode==0 and job.isdigit():out['job_id']=job
except subprocess.TimeoutExpired:out=dict(outcome='unknown_timeout_inspect_do_not_resubmit',attempt=attempt)
once(root/(kind+'_submission.json'),out)
tmp=root/(phase+'_submission.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');os.replace(tmp,active)
if 'job_id' in out:
 p=subprocess.run(['scontrol','release',out['job_id']],capture_output=True,text=True,timeout=20)
 release=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 once(root/(kind+'_release.json'),release);out={**out,'release':release}
print(json.dumps(out))
'''


def submit(phase, resume=False):
    reg = register()
    for rel in [*reg['bindings'], str(REG.relative_to(base.ROOT))]:
        q = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=base.ROOT, capture_output=True)
        if q.returncode or q.stdout != (base.ROOT/rel).read_bytes():
            raise ValueError('Commit registered code before execution: '+rel)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for rel in reg['bindings']:
            raw = (base.ROOT/rel).read_bytes(); entry = tarfile.TarInfo(rel); entry.size = len(raw)
            tar.addfile(entry, io.BytesIO(raw))
    installed = base.remote(INSTALL, [REMOTE, base.REMOTE, REG.read_text(), CONFIG.read_text()], buf.getvalue(), timeout=90)
    result = base.remote(SUBMIT, [REMOTE, base.RUNTIME, phase, base.sha(REG), 'resume' if resume else 'new'], timeout=90)
    base.once(PUBLIC/(phase+'_attempt'+str(result['attempt'])+'_submission.json'), result)
    return dict(installation=installed, submission=result)


OBSERVE = r'''
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];collect=sys.argv[3]=='collect'
assert root==pathlib.Path('/users/k24101830/m3w/european_task_risk_v1')
sys.path.insert(0,str(root/'code'))
from scripts.train_m3w_task_risk import verify,PUBLIC,PRIVATE,sha
from scripts.manage_m3w_task_risk import controller_success
verify(root)
sub=json.loads((root/(phase+'_submission.json')).read_text());job=sub['job_id'];assert job.isdigit()
p=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=20)
out=dict(submission=sub,controller=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
for suffix in ('out','err'):
 path=root/(phase+'-'+job+'.'+suffix)
 if path.exists():
  with path.open('rb') as f:f.seek(max(0,path.stat().st_size-3000));out[suffix]=f.read().decode(errors='replace')
hb=root/PRIVATE/(phase+'_heartbeat.json');out['heartbeat']=json.loads(hb.read_text()) if hb.exists() else None
name='pilot.json' if phase=='pilot' else 'training_freeze.json';home=root/PUBLIC;path=home/name
out['artifact']=json.loads(path.read_text()) if path.exists() else None
if collect:
 assert controller_success(out['controller'],job,phase),'Actual successful controller exit required'
 assert out['artifact'] and out['artifact']['job_id']==job
 assert out['artifact']['registration_sha256']==sha(root/'registration.json')
 out['files']={name:path.read_text()};out['checkpoint_payloads_verified']=0
 if phase=='train':
  assert out['artifact']['neural_fits']==216 and len(out['artifact']['fits'])==216
  for ref in out['artifact']['fits']:
   path=(root/ref['path']).resolve();assert path.is_relative_to((home/'fits').resolve()) and sha(path)==ref['sha256']
   raw=path.read_text();doc=json.loads(raw);cp=doc['checkpoint'];p=(root/cp['path']).resolve()
   assert p.is_relative_to((root/PRIVATE/'heads').resolve()) and not p.is_symlink()
   assert p.stat().st_size==cp['bytes'] and sha(p)==cp['sha256'] and doc['step']==2000
   assert doc['registration_sha256']==sha(root/'registration.json') and not doc['validation_scored']
   out['checkpoint_payloads_verified']+=1;out['files'][str(path.relative_to(home))]=raw
 assert sum(len(v.encode()) for v in out['files'].values())<8*2**20
print(json.dumps(out))
'''


def observe(phase, collect=False):
    out = base.remote(OBSERVE, [REMOTE, phase, 'collect' if collect else 'status'], timeout=90)
    if collect:
        for rel, raw in out.pop('files').items():
            p = (PUBLIC/rel).resolve()
            if not p.is_relative_to(PUBLIC.resolve()): raise ValueError('Owned metadata only')
            base.once(p, json.loads(raw))
        base.once(PUBLIC/(phase+'_collection.json'), out)
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        base.once(PUBLIC/(phase+'_observation_'+stamp+'.json'), out)
    if out.get('artifact') and 'fits' in out['artifact']:
        out = {**out, 'artifact':{k:v for k,v in out['artifact'].items() if k != 'fits'}}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['register', 'pilot', 'train', 'status', 'collect', 'resume'])
    p.add_argument('--phase', choices=['pilot', 'train'], default='pilot'); a = p.parse_args()
    if a.mode == 'register':
        r = register(); out = dict(registration_sha256=base.sha(REG), bindings=len(r['bindings']))
    elif a.mode in ('pilot', 'train', 'resume'):
        out = submit(a.phase if a.mode == 'resume' else a.mode, resume=a.mode == 'resume')
    else:
        out = observe(a.phase, collect=a.mode == 'collect')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
