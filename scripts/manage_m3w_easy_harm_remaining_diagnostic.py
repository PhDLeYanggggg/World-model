"""Register and run the remaining-control diagnostic without modifying fits."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

from scripts import manage_m3w_easy_harm_deviance as manager
from scripts.diagnose_m3w_easy_harm_remaining_controls import remaining_refs, NAME

ROOT = manager.base.ROOT
HOME = manager.PUBLIC
REG = HOME/(NAME+'_registration.json')
OBS = HOME/'partial_training_observation_20261006T1256Z.json'
SCRIPT = ROOT/'scripts/diagnose_m3w_easy_harm_remaining_controls.py'


def register():
    manifest = json.loads((manager.base.PRIVATE/'create_train_input_manifest.json').read_text())
    refs = remaining_refs(manifest, json.loads(OBS.read_text()))
    if len(refs) != 17:
        raise ValueError('Observed 17-pair diagnostic only')
    bindings = {str(p.relative_to(ROOT)): manager.base.sha(p) for p in
                (SCRIPT, ROOT/'scripts/diagnose_m3w_easy_harm_control.py')}
    value = dict(training_registration_sha256=manager.base.sha(manager.REG),
        code_bindings=bindings, observation_sha256=manager.base.sha(OBS),
        manager_sha256=manager.base.sha(__file__),
        test_sha256=manager.base.sha(ROOT/'tests/test_m3w_easy_harm_remaining_diagnostic.py'),
        identities=[manager.run.key(r, 'quadratic') for r in refs],
        protocol='17 unaccepted TRAIN identities; original and new direct2000-step fits; compare saved and historical states; no acceptance waiver',
        diagnostic_node='erc-hpc-comp186', verification_updates=68000,
        scientific_training_budget_changed=False, validation_scored=False, independent_roles_read=False)
    manager.base.once(REG, value)
    return value


SUBMIT = r'''
import hashlib,io,json,pathlib,subprocess,sys,tarfile
root=pathlib.Path(sys.argv[1]);regraw=sys.argv[2];reg=json.loads(regraw);name='control_diagnostic_v2'
assert root==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
assert json.loads((root/'.owner.json').read_text())['registration_sha256']==reg['training_registration_sha256']
sys.path.insert(0,str(root/'code'))
from scripts.train_m3w_easy_harm_deviance import verify,quota,once,PUBLIC
r,source,cfg,_,_=verify(root);resources=quota(r,source,cfg)
assert not (root/(name+'_submission_intent.json')).exists()
allowed={**reg['code_bindings'],str(PUBLIC/'partial_training_observation_20261006T1256Z.json'):reg['observation_sha256']}
seen=set()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for entry in tar.getmembers():
  assert entry.isfile() and entry.name in allowed and entry.name not in seen
  raw=tar.extractfile(entry).read();assert hashlib.sha256(raw).hexdigest()==allowed[entry.name]
  path=root/entry.name if entry.name.startswith('outputs/') else root/'code'/entry.name
  assert path.resolve().is_relative_to(root.resolve());path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():assert path.read_bytes()==raw
  else:
   with path.open('xb') as f:f.write(raw)
  seen.add(entry.name)
assert seen==set(allowed)
once(root/(name+'_registration.json'),reg)
runtime='/users/k24101830/m3w/easy_hurdle_runtime_v2/venv/bin/python'
batch=root/(name+'.sbatch')
script='\n'.join(['#!/bin/bash -l','#SBATCH --job-name=m3w_easy_remaining_diagnostic',
'#SBATCH --account=kcl','#SBATCH --partition=cpu','#SBATCH --qos=normal',
'#SBATCH --ntasks=1','#SBATCH --cpus-per-task=4','#SBATCH --mem=16G',
'#SBATCH --time=02:00:00','#SBATCH --nodelist='+reg['diagnostic_node'],
'#SBATCH --output='+str(root)+'/remaining-diagnostic-%j.out',
'#SBATCH --error='+str(root)+'/remaining-diagnostic-%j.err','set -euo pipefail',
'module load python/3.11.6-gcc-13.2.0','unset PYTHONPATH',
'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4',
'cd '+str(root/'code'),'exec '+runtime+' -m scripts.diagnose_m3w_easy_harm_remaining_controls --root '+str(root),''])
with batch.open('x') as f:f.write(script)
once(root/(name+'_submission_intent.json'),dict(resources=resources,registration_sha256=hashlib.sha256(regraw.encode()).hexdigest()))
try:
 p=subprocess.run(['sbatch','--hold','--parsable',str(batch)],capture_output=True,text=True,timeout=30)
 out=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 job=p.stdout.strip().split(';')[0]
 if p.returncode==0 and job.isdigit():out['job_id']=job
except subprocess.TimeoutExpired:out=dict(outcome='unknown_timeout_inspect_do_not_resubmit')
once(root/(name+'_submission.json'),out)
if 'job_id' in out:
 q=subprocess.run(['scontrol','release',out['job_id']],capture_output=True,text=True,timeout=20)
 out['release']=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
 once(root/(name+'_release.json'),out['release'])
print(json.dumps(out))
'''

OBSERVE = r'''
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);assert root==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
name='control_diagnostic_v2';assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
out={'submission':json.loads((root/(name+'_submission.json')).read_text())};job=out['submission']['job_id'];assert job.isdigit()
for key,cmd in [('queue',['squeue','-j',job,'-h','-o','%i|%T|%M|%R']),('accounting',['sacct','-j',job,'-X','--noheader','--parsable2','--format=JobID,State,Elapsed,ExitCode,NodeList'])]:
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=20);out[key]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
for ext in ('out','err'):
 path=root/('remaining-diagnostic-'+job+'.'+ext);out[ext]=path.read_text()[-5000:] if path.exists() else None
home=root/'outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1'/name
out['complete']=None;out['rows']=[];path=home/'complete.json'
if path.exists():
 out['complete']=json.loads(path.read_text());out['complete_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 for ref in out['complete']['results']:
  p=(root/ref['path']).resolve();assert p.is_relative_to(home.resolve()) and p.stat().st_size<2*2**20
  assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256'];out['rows'].append({'path':ref['path'],'value':json.loads(p.read_text())})
print(json.dumps(out))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register','submit','status','collect'])
    args = p.parse_args()
    if args.phase == 'register':
        print(json.dumps(register(), indent=2)); return
    if args.phase == 'submit':
        reg = register()
        files = [ROOT/p for p in reg['code_bindings']] + [OBS]
        for path in files+[REG,Path(__file__).resolve(),ROOT/'tests/test_m3w_easy_harm_remaining_diagnostic.py']:
            assert subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT) == path.read_bytes()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf,mode='w:gz') as tar:
            for path in files:
                raw = path.read_bytes(); entry = tarfile.TarInfo(str(path.relative_to(ROOT)))
                entry.size = len(raw); tar.addfile(entry,io.BytesIO(raw))
        out = manager.base.remote(SUBMIT,[manager.REMOTE,REG.read_text()],buf.getvalue(),timeout=90)
        manager.base.once(HOME/(NAME+'_submission.json'),out)
    else:
        out = manager.base.remote(OBSERVE,[manager.REMOTE],timeout=90)
        if args.phase == 'collect':
            assert out['complete'] and out['submission']['job_id']+'|COMPLETED|' in out['accounting']['stdout']
            assert '|0:0|' in out['accounting']['stdout']
            assert out['complete']['registration_sha256'] == manager.base.sha(REG)
            for row in out['rows']:
                manager.base.once(ROOT/row['path'],row['value'])
            manager.base.once(HOME/NAME/'complete.json',out['complete'])
            assert manager.base.sha(HOME/NAME/'complete.json') == out['complete_sha256']
            manager.base.once(HOME/(NAME+'_collection.json'),out)
    if args.phase in ('status','collect') and out['complete']:
        print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
    else:
        print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
