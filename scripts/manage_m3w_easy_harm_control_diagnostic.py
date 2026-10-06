"""One owned compute diagnostic; never change the frozen training acceptance rule."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts import manage_m3w_easy_harm_deviance as manager

SCRIPT = manager.base.ROOT/'scripts/diagnose_m3w_easy_harm_control.py'
HOME = manager.PUBLIC
REG = HOME/'control_diagnostic_registration.json'


def registration():
    value = dict(training_registration_sha256=manager.base.sha(manager.REG),
        script_sha256=manager.base.sha(SCRIPT),
        protocol='first TRAIN identity; unchanged 2000-step original and new quadratic direct fits; original100-step pilot resume; compare historical and failed states',
        diagnostic_node='erc-hpc-comp186', primary_acceptance_unchanged=True,
        candidate_hypothesis='cross-node floating-point reduction differences, unproven',
        test_bindings={'tests/test_m3w_easy_harm_control_diagnostic.py':manager.base.sha(
            manager.base.ROOT/'tests/test_m3w_easy_harm_control_diagnostic.py')},
        validation_scored=False, independent_roles_read=False)
    manager.base.once(REG, value)
    return value


SUBMIT = r'''
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);regraw=sys.argv[2];reg=json.loads(regraw)
assert root==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
assert json.loads((root/'.owner.json').read_text())['registration_sha256']==reg['training_registration_sha256']
sys.path.insert(0,str(root/'code'))
from scripts.train_m3w_easy_harm_deviance import verify,quota,once,PUBLIC
r,source,cfg,_,_=verify(root);resources=quota(r,source,cfg)
assert not (root/'control_diagnostic_submission_intent.json').exists()
raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==reg['script_sha256']
path=root/'code/scripts/diagnose_m3w_easy_harm_control.py'
if path.exists():assert path.read_bytes()==raw
else:
 with path.open('xb') as f:f.write(raw)
once(root/'control_diagnostic_registration.json',reg)
runtime='/users/k24101830/m3w/easy_hurdle_runtime_v2/venv/bin/python'
batch=root/'control_diagnostic.sbatch'
script='\n'.join(['#!/bin/bash -l','#SBATCH --job-name=m3w_easy_control_diagnostic',
'#SBATCH --account=kcl','#SBATCH --partition=cpu','#SBATCH --qos=normal',
'#SBATCH --ntasks=1','#SBATCH --cpus-per-task=4','#SBATCH --mem=16G',
'#SBATCH --time=02:00:00','#SBATCH --nodelist='+reg['diagnostic_node'],
'#SBATCH --output='+str(root)+'/control-diagnostic-%j.out',
'#SBATCH --error='+str(root)+'/control-diagnostic-%j.err','set -euo pipefail',
'module load python/3.11.6-gcc-13.2.0','unset PYTHONPATH',
'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4',
'cd '+str(root/'code'),'exec '+runtime+' -m scripts.diagnose_m3w_easy_harm_control --root '+str(root),''])
with batch.open('x') as f:f.write(script)
once(root/'control_diagnostic_submission_intent.json',dict(resources=resources,registration_sha256=hashlib.sha256(regraw.encode()).hexdigest()))
p=subprocess.run(['sbatch','--hold','--parsable',str(batch)],capture_output=True,text=True,timeout=30)
out=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
job=p.stdout.strip().split(';')[0]
if p.returncode==0 and job.isdigit():out['job_id']=job
once(root/'control_diagnostic_submission.json',out)
if 'job_id' in out:
 q=subprocess.run(['scontrol','release',job],capture_output=True,text=True,timeout=20)
 out['release']=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
 once(root/'control_diagnostic_release.json',out['release'])
print(json.dumps(out))
'''


OBSERVE = r'''
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);assert root==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
out={'submission':json.loads((root/'control_diagnostic_submission.json').read_text())};job=out['submission']['job_id'];assert job.isdigit()
for key,cmd in [('queue',['squeue','-j',job,'-h','-o','%i|%T|%M|%R']),('accounting',['sacct','-j',job,'-X','--noheader','--parsable2','--format=JobID,State,Elapsed,ExitCode,NodeList'])]:
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=20);out[key]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
for ext in ('out','err'):
 path=root/('control-diagnostic-'+job+'.'+ext);out[ext]=path.read_text()[-3000:] if path.exists() else None
path=root/'outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1/control_diagnostic_v1.json'
out['result']=None
if path.exists():
 assert path.stat().st_size<2*2**20
 out['result']=json.loads(path.read_text());out['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps(out))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['register', 'submit', 'status', 'collect'])
    args = parser.parse_args()
    if args.phase == 'register':
        print(json.dumps(registration(), indent=2)); return
    if args.phase == 'submit':
        registration()
        for path in (SCRIPT, REG, Path(__file__).resolve()):
            saved = subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(manager.base.ROOT))], cwd=manager.base.ROOT)
            assert saved == path.read_bytes(), 'Commit diagnosis before execution'
        out = manager.base.remote(SUBMIT,[manager.REMOTE,REG.read_text()],SCRIPT.read_bytes(),timeout=90)
        manager.base.once(HOME/'control_diagnostic_submission.json',out)
    else:
        out = manager.base.remote(OBSERVE,[manager.REMOTE],timeout=90)
        if args.phase == 'collect':
            assert out['result'] and out['submission']['job_id']+'|COMPLETED|' in out['accounting']['stdout']
            assert out['result']['script_sha256'] == manager.base.sha(SCRIPT)
            manager.base.once(HOME/'control_diagnostic_v1.json',out['result'])
            assert manager.base.sha(HOME/'control_diagnostic_v1.json') == out['sha256']
            manager.base.once(HOME/'control_diagnostic_collection.json',out)
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
