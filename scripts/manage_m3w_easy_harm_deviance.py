"""Register, submit and inspect the owned paired easy-harm loss experiment."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

from scripts import manage_m3w_temporal_auxiliary_create as base
from scripts import train_m3w_easy_harm_deviance as run

PUBLIC = base.ROOT/run.PUBLIC
CONFIG = base.ROOT/'configs'/('m3w_'+run.NAME+'.json')
REG = PUBLIC/'registration.json'
REMOTE = '/users/k24101830/m3w/'+run.NAME


def register():
    from scripts.export_m3w_easy_hurdle_create import closure
    cfg = json.loads(CONFIG.read_text())
    manifest = json.loads((base.PRIVATE/'create_train_input_manifest.json').read_text())
    files = closure(base.ROOT,['scripts.train_m3w_easy_harm_deviance'])
    files += [Path(__file__).resolve(), CONFIG, PUBLIC/'protocol.md',
              base.ROOT/'tests/test_m3w_easy_harm_deviance_training.py',
              base.ROOT/'tests/test_m3w_easy_harm_deviance.py',
              base.ROOT/'tests/test_m3w_easy_harm_deviance_execution.py']
    assert cfg['head_training'] == json.loads(base.CONFIG.read_text())['head_training']
    assert len(manifest['heads']) == 72 and cfg['risk_budget'] == .02
    value = dict(experiment=run.NAME,parent=run.PARENT,config_sha256=base.sha(CONFIG),
        input_manifest_sha256=base.sha(base.PRIVATE/'create_train_input_manifest.json'),
        parent_training_freeze_sha256=base.sha(base.PUBLIC/'training_freeze.json'),
        parent_diagnostic_sha256=base.sha(base.PUBLIC/'false_safe_train_diagnostic_v1/summary.json'),
        parent_registration_sha256=base.sha(base.PUBLIC/'registration.json'),
        bindings={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(set(files))},
        expected=[dict(group=r['group'],source=r['identity']['source'],seed=r['identity']['seed']) for r in manifest['heads']],
        assignment=[[run.key(r,a) for r in manifest['heads'][i::4] for a in cfg['arms']] for i in range(4)],
        pilot_identity=manifest['heads'][0]['identity'], source_packets_copied=False,
        primary_changed_term='fifth moment quadratic to normalized easy-harm cost deviance only',
        full_fits=144, source_identities=72, steps=2000, checkpoint_selection=False,
        threshold_search=False, risk_budget=.02, independent_roles_read=False, new_forecaster=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    base.once(REG,value)
    return value


INSTALL = r'''
import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]);source=pathlib.Path(sys.argv[2]);regraw=sys.argv[3];cfgraw=sys.argv[4]
reg=json.loads(regraw);assert root==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
assert source==pathlib.Path('/users/k24101830/m3w/european_temporal_auxiliary_v1')
assert json.loads((source/'.owner.json').read_text())['registration_sha256']==reg['parent_registration_sha256']
assert hashlib.sha256((source/'train_input_manifest.json').read_bytes()).hexdigest()==reg['input_manifest_sha256']
assert hashlib.sha256(cfgraw.encode()).hexdigest()==reg['config_sha256']
root.mkdir(exist_ok=True)
def exact(path,raw):
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
  p=(root/'code'/m.name);assert p.resolve().is_relative_to((root/'code').resolve())
  p.parent.mkdir(parents=True,exist_ok=True);exact(p,raw);seen[m.name]=h
assert seen==reg['bindings']
print(json.dumps(dict(code_bindings=len(seen),source_packets_copied=False,source_modified=False)))
'''


SUBMIT = r'''
import json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);runtime=sys.argv[2];phase=sys.argv[3];expected=sys.argv[4]
sys.path.insert(0,str(root/'code'))
from scripts.train_m3w_easy_harm_deviance import verify,quota,sha,once,PUBLIC
root,source,cfg,reg,manifest=verify(root);assert sha(root/'registration.json')==expected
assert phase in ('pilot','train')
assert not (root/(phase+'_submission_intent.json')).exists(),'Inspect existing intent; no duplicate submission'
resources=quota(root,source,cfg)
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%P|%T|%C'],capture_output=True,text=True,timeout=20);assert q.returncode==0
assert not any('|m3w_easy_deviance_' in s for s in q.stdout.splitlines())
if phase=='train':
 pilot=json.loads((root/PUBLIC/'pilot.json').read_text());assert pilot['engineering_pass']
 sub=json.loads((root/'pilot_submission.json').read_text());assert sub['job_id']==pilot['job_id']
 a=subprocess.run(['sacct','-j',sub['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
 assert a.returncode==0 and a.stdout.strip()=='COMPLETED|0:0'
 assert not (root/PUBLIC/'training_freeze.json').exists()
def script(kind):
 join=kind=='join';cpu=1 if join else 4;mem='2G' if join else '16G';wall='00:20:00' if join else '02:00:00' if kind=='pilot' else '12:00:00'
 log='%A_%a' if kind=='train' else '%j'
 cmd=f'exec {runtime}/venv/bin/python -m scripts.train_m3w_easy_harm_deviance {kind} --root {root}'
 if kind=='train':cmd+=' --shard "$SLURM_ARRAY_TASK_ID"'
 return '\n'.join(['#!/bin/bash -l',f'#SBATCH --job-name=m3w_easy_deviance_{kind}','#SBATCH --account=kcl',
 '#SBATCH --partition=cpu','#SBATCH --qos=normal','#SBATCH --ntasks=1',f'#SBATCH --cpus-per-task={cpu}',
 f'#SBATCH --mem={mem}',f'#SBATCH --time={wall}',f'#SBATCH --output={root}/{kind}-{log}.out',
 f'#SBATCH --error={root}/{kind}-{log}.err','#SBATCH --signal=B:TERM@120','set -euo pipefail',
 'module load python/3.11.6-gcc-13.2.0','unset PYTHONPATH',
 f'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS={cpu} OPENBLAS_NUM_THREADS={cpu} MKL_NUM_THREADS={cpu} NUMEXPR_NUM_THREADS={cpu}',
 f'cd {root}/code',cmd,''])
def dispatch(kind,args):
 path=root/(kind+'.sbatch')
 with path.open('x') as f:f.write(script(kind))
 once(root/(kind+'_submission_intent.json'),dict(registration_sha256=expected,resources=resources,
  m3w_queue=[s for s in q.stdout.splitlines() if '|m3w_' in s],other_jobs_untouched=True))
 try:
  p=subprocess.run(['sbatch','--parsable','--hold',*args,str(path)],capture_output=True,text=True,timeout=30)
  out=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,registration_sha256=expected)
  job=p.stdout.strip().split(';')[0]
  if p.returncode==0 and job.isdigit():out['job_id']=job
 except subprocess.TimeoutExpired:out=dict(outcome='unknown_timeout_inspect_do_not_resubmit')
 once(root/(kind+'_submission.json'),out);return out
out={phase:dispatch(phase,[] if phase=='pilot' else ['--array=0-3%4'])}
if 'job_id' in out[phase]:
 if phase=='train':out['join']=dispatch('join',['--dependency=afterok:'+out['train']['job_id']])
 if phase=='pilot' or 'job_id' in out.get('join',{}):
  out['release']=[]
  for kind in [phase]+(['join'] if phase=='train' else []):
   p=subprocess.run(['scontrol','release',out[kind]['job_id']],capture_output=True,text=True,timeout=20)
   out['release'].append(dict(kind=kind,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
  once(root/(phase+'_release.json'),out['release'])
print(json.dumps(out))
'''


def submit(phase):
    reg=register()
    for rel in [*reg['bindings'],str(REG.relative_to(base.ROOT))]:
        q=subprocess.run(['git','show','HEAD:'+rel],cwd=base.ROOT,capture_output=True)
        assert q.returncode==0 and q.stdout==(base.ROOT/rel).read_bytes(),'Commit before execution'
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tar:
        for rel in reg['bindings']:
            raw=(base.ROOT/rel).read_bytes();entry=tarfile.TarInfo(rel);entry.size=len(raw);tar.addfile(entry,io.BytesIO(raw))
    installed=base.remote(INSTALL,[REMOTE,base.REMOTE,REG.read_text(),CONFIG.read_text()],buf.getvalue(),timeout=90)
    result=base.remote(SUBMIT,[REMOTE,base.RUNTIME,phase,base.sha(REG)],timeout=90)
    base.once(PUBLIC/(phase+'_submission.json'),result)
    return dict(installation=installed,submission=result)


OBSERVE = r'''
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];collect=sys.argv[3]=='collect'
assert root==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
out={};home=root/'outputs/publication_readiness_2026_09'/root.name
for kind in [phase]+(['join'] if phase=='train' else []):
 sub=json.loads((root/(kind+'_submission.json')).read_text());job=sub['job_id'];assert job.isdigit()
 row={'submission':sub}
 for key,cmd in [('queue',['squeue','-j',job,'-h','-o','%i|%T|%M|%R']),('accounting',['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,Elapsed,ExitCode,MaxRSS'])]:
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=20);row[key]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 if kind!='train':
  for suffix in ('out','err'):
   p=root/(kind+'-'+job+'.'+suffix);row[suffix]=p.read_text()[-2000:] if p.exists() else None
 out[kind]=row
out['heartbeats']={}
for tag in (['pilot'] if phase=='pilot' else ['shard'+str(i) for i in range(4)]):
 p=root/'data/stage_cvpr2027_experiments'/root.name/tag/'heartbeat.json'
 out['heartbeats'][tag]=json.loads(p.read_text()) if p.exists() else None
artifact='pilot.json' if phase=='pilot' else 'training_freeze.json';p=home/artifact
out['artifact']=json.loads(p.read_text()) if p.exists() else None
if collect:
 terminal=out['pilot' if phase=='pilot' else 'join'];job=terminal['submission']['job_id']
 assert out['artifact'] and job+'|COMPLETED|' in terminal['accounting']['stdout']
 out['files']={artifact:p.read_text()}
 if phase=='train':
  for ref in out['artifact']['fits']:
   path=(root/ref['path']).resolve();assert path.is_relative_to(home.resolve()/'fits')
   raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];out['files'][str(path.relative_to(home.resolve()))]=raw.decode()
 assert sum(len(v.encode()) for v in out['files'].values())<8*2**20
print(json.dumps(out))
'''


def observe(phase,collect):
    out=base.remote(OBSERVE,[REMOTE,phase,'collect' if collect else 'status'],timeout=90)
    if collect:
        for rel, text in out.pop('files').items():
            path=(PUBLIC/rel).resolve();assert path.is_relative_to(PUBLIC.resolve())
            base.once(path,json.loads(text))
        base.once(PUBLIC/(phase+'_collection.json'),out)
    if out.get('artifact') and 'fits' in out['artifact']:
        out={**out,'artifact':{k:v for k,v in out['artifact'].items() if k!='fits'}}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','pilot','train','status','collect'])
    p.add_argument('--phase',choices=['pilot','train'],default='pilot');a=p.parse_args()
    if a.mode=='register':
        reg=register();out=dict(registration_sha256=base.sha(REG),bindings=len(reg['bindings']))
    elif a.mode in ('pilot','train'):out=submit(a.mode)
    else:out=observe(a.phase,a.mode=='collect')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
