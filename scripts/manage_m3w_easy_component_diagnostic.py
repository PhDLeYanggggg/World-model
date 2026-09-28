"""Register and manage immutable fitting-only component jobs on CREATE."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.manage_m3w_easy_gradient_diagnostic import call_remote, digest, immutable
from scripts.export_m3w_easy_hurdle_create import closure

NAME = 'european_easy_component_diagnostic_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
PARENT = PUBLIC.parent/'european_easy_risk_priority_v1'
REMOTE = '/users/k24101830/m3w/'+NAME
PACKETS = '/users/k24101830/m3w/european_easy_hurdle_v1'
TRAINED = '/users/k24101830/m3w/european_easy_risk_priority_v1'


def registration():
    cfg = json.loads(CONFIG.read_text()); seal = json.loads((PARENT/'verification.json').read_text())
    assert digest(PARENT/'verification.json') == cfg['parent_seal_sha256']
    for rel, sha in seal['source_bindings'].items(): assert digest(ROOT/rel) == sha
    for rel, sha in seal['artifacts'].items(): assert digest(PARENT/rel) == sha
    previous = json.loads((ROOT/'configs/m3w_european_easy_risk_priority_v1.json').read_text())
    assert cfg['parent_manifest_sha256'] == previous['parent_manifest_sha256']
    paths = closure(ROOT, ['scripts.run_m3w_easy_component_diagnostic'])
    controls = [CONFIG, PUBLIC/'protocol.md', Path(__file__), ROOT/'tests/test_m3w_easy_component_diagnostic.py']
    return dict(name=NAME, parent_seal_sha256=cfg['parent_seal_sha256'],
        code_bindings={str(p.relative_to(ROOT)): digest(p) for p in paths},
        control_bindings={str(p.relative_to(ROOT)): digest(p) for p in controls},
        groups=108, checkpoints=216, parameter_updates=0, held_outcomes_used=False,
        independent_roles_read=False, policy_actions_computed=False)


def submit(reg, replay=False):
    label = 'replay' if replay else 'diagnose'; record = PRIVATE/('submission_'+label+'.json')
    assert not record.exists(), 'Inspect previous/uncertain submission; never duplicate'
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', str(PUBLIC/'registration.json')], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(['git', 'cat-file', '-e', 'HEAD:'+str((PUBLIC/'registration.json').relative_to(ROOT))], cwd=ROOT, check=True)
    flag = '--replay' if replay else ''
    script = f'''#!/bin/bash -l
#SBATCH --job-name=m3w_easy_component_{label}
#SBATCH --partition=cpu
#SBATCH --account=kcl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output={label}-%j.out
#SBATCH --error={label}-%j.err
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
module load python/3.11.6-gcc-13.2.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset PYTHONPATH
../easy_hurdle_runtime_v2/venv/bin/python code/scripts/run_m3w_easy_component_diagnostic.py --home "$PWD" --parent-home {PACKETS} --trained-home {TRAINED} {flag}
'''
    payload = dict(home=REMOTE, parent=PACKETS, trained=TRAINED, name=NAME, label=label,
        config=json.loads(CONFIG.read_text()), registration=reg, script=script,
        files={rel: (ROOT/rel).read_text() for rel in reg['code_bindings']})
    code = r'''
import hashlib,json,os,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['home']);parent=pathlib.Path(p['parent']);trained=pathlib.Path(p['trained'])
assert root.parent==parent.parent==trained.parent and root.name==p['name']
assert json.loads((root.parent/'.m3w_owner.json').read_text())['project']=='M3W'
assert hashlib.sha256((parent/'input_manifest.json').read_bytes()).hexdigest()==p['config']['parent_manifest_sha256']
assert hashlib.sha256((trained/'training_complete.json').read_bytes()).hexdigest()==p['config']['trained_receipt_sha256']
prior=json.loads((trained/'training_complete.json').read_text())
a=subprocess.run(['sacct','-j',prior['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
assert a.returncode==0 and a.stdout.strip()=='COMPLETED|0:0'
limit=int(os.getxattr('/users/k24101830','ceph.quota.max_bytes'));used=int(os.getxattr('/users/k24101830','ceph.dir.rbytes'))
assert limit>used+256*2**20,'Preserve personal quota and old artifacts'
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and 'm3w_easy_component_' not in q.stdout
root.mkdir(exist_ok=True)
def once(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==text
    else:
        with path.open('x') as f:f.write(text)
once(root/'.owner.json',json.dumps({'project':'M3W','experiment':p['name']})+'\n')
intent=root/('submit_intent_'+p['label']+'.json');receipt=root/('submit_receipt_'+p['label']+'.json')
assert not intent.exists() and not receipt.exists(),'Prior or uncertain submission'
if p['label']=='replay':
    done=json.loads((root/'receipt.json').read_text());assert done['groups']==108
    a=subprocess.run(['sacct','-j',done['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
    assert a.returncode==0 and a.stdout.strip()=='COMPLETED|0:0'
for name,value in [('config.json',p['config']),('registration.json',p['registration'])]:
    once(root/name,json.dumps(value,indent=2,allow_nan=False)+'\n')
for name,text in p['files'].items():
    assert not pathlib.PurePosixPath(name).is_absolute() and '..' not in pathlib.PurePosixPath(name).parts
    assert hashlib.sha256(text.encode()).hexdigest()==p['registration']['code_bindings'][name]
    once(root/'code'/name,text)
script=root/('run_'+p['label']+'.sh');once(script,p['script'])
once(intent,json.dumps({'script_sha256':hashlib.sha256(p['script'].encode()).hexdigest(),'quota_limit_bytes':limit,'quota_used_bytes':used})+'\n')
try:
    r=subprocess.run(['sbatch','--parsable',script.name],cwd=root,capture_output=True,text=True,timeout=30)
    out=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    if r.returncode==0:
        out['job_id']=r.stdout.strip().split(';')[0];assert out['job_id'].isdigit()
    once(receipt,json.dumps(out,indent=2)+'\n');print(json.dumps(out))
except subprocess.TimeoutExpired:
    print(json.dumps({'returncode':None,'uncertain':True,'intent_retained':True}))
'''
    result = call_remote(code, payload)
    immutable(record, dict(utc=datetime.now(timezone.utc).isoformat(), label=label, result=result))
    response = json.loads(result['stdout']) if result.get('returncode') == 0 else {}
    print(json.dumps(dict(transport_returncode=result['returncode'], response=response,
        uncertain=result['returncode'] != 0 or response.get('returncode') is None, record=str(record))))


def inspect(collect=False):
    code = r'''
import hashlib,json,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['home']);result={}
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T|%R'],capture_output=True,text=True,timeout=20)
result['queue']=dict(returncode=q.returncode,stdout=q.stdout)
for label in ('diagnose','replay'):
    result[label+'_intent']=(root/('submit_intent_'+label+'.json')).exists()
    path=root/('submit_receipt_'+label+'.json')
    if path.exists():
        d=json.loads(path.read_text());result[label+'_submission']=d
        if d.get('job_id'):
            a=subprocess.run(['sacct','-j',d['job_id'],'--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed,MaxRSS'],capture_output=True,text=True,timeout=20)
            result[label+'_accounting']=dict(returncode=a.returncode,stdout=a.stdout)
            err=root/(label+'-'+d['job_id']+'.err')
            if err.exists():result[label+'_stderr_tail']=err.read_text()[-3000:]
for name in ('heartbeat.json','receipt.json','replay_receipt.json'):
    if (root/name).exists():result[name]=json.loads((root/name).read_text())
if p['collect']:
    for name in ('receipt.json','replay_receipt.json'):
        doc=result[name]
        a=subprocess.run(['sacct','-j',doc['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
        assert a.returncode==0 and a.stdout.strip()=='COMPLETED|0:0'
        assert doc['groups']==108 and doc['parameter_updates']==0 and not doc['held_outcomes_used']
    assert result['replay_receipt.json']['replay_exact']
    assert result['receipt.json']['artifacts']==result['replay_receipt.json']['artifacts']
    rows=[]
    for ref in result['receipt.json']['artifacts']:
        path=root/ref['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
        rows.append(json.loads(path.read_text()))
    result['groups']=rows
print(json.dumps(result))
'''
    result = call_remote(code, dict(home=REMOTE, collect=collect))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    immutable(PRIVATE/('observation_'+stamp+'.json'), result)
    if result.get('returncode') != 0:
        print(json.dumps(dict(status='observation_unavailable_not_terminal', returncode=result['returncode']))); return
    obs = json.loads(result['stdout'])
    if collect:
        assert len(obs['groups']) == 108
        immutable(PRIVATE/'collected.json', obs)
        print(json.dumps(dict(collected_groups=108, replay_exact=True)))
    else:
        compact = {k: v for k, v in obs.items() if k not in ('receipt.json', 'replay_receipt.json')}
        for key in ('receipt.json', 'replay_receipt.json'):
            if key in obs: compact[key] = {k: v for k, v in obs[key].items() if k != 'artifacts'}
        print(json.dumps(compact, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'submit', 'submit-replay', 'inspect', 'collect']); a = p.parse_args()
    reg = registration()
    if a.phase == 'register':
        immutable(PUBLIC/'registration.json', reg); print(json.dumps(dict(registered=True, files=len(reg['code_bindings']))))
    else:
        assert json.loads((PUBLIC/'registration.json').read_text()) == reg
        if a.phase.startswith('submit'): submit(reg, a.phase == 'submit-replay')
        else: inspect(a.phase == 'collect')


if __name__ == '__main__': main()
