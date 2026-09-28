"""Hash-bound registration and bounded CREATE operations for fitting diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_m3w_easy_hurdle_create import closure

NAME = 'european_easy_gradient_diagnostic_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
PARENT = ROOT/'outputs/publication_readiness_2026_09/european_easy_hurdle_v1'
CONFIG = ROOT/'configs/m3w_european_easy_gradient_diagnostic_v1.json'
HANDOFF = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'
REMOTE_PARENT = '/users/k24101830/m3w/european_easy_hurdle_v1'
REMOTE = '/users/k24101830/m3w/'+NAME


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == raw
    else:
        with path.open('x') as f:
            f.write(raw)


def registration():
    cfg = json.loads(CONFIG.read_text()); seal = PARENT/'verification.json'
    assert digest(seal) == cfg['parent_seal_sha256']
    doc = json.loads(seal.read_text())
    for p, h in doc['source_bindings'].items():
        assert digest(ROOT/p) == h
    for p, h in doc['artifacts'].items():
        assert digest(PARENT/p) == h
    paths = closure(ROOT, ['scripts.run_m3w_easy_gradient_diagnostic'])
    controls = [CONFIG, PUBLIC/'protocol.md', Path(__file__), ROOT/'tests/test_m3w_easy_gradient_diagnostic.py']
    return dict(name=NAME, parent_seal_sha256=digest(seal),
        code_bindings={str(p.relative_to(ROOT)): digest(p) for p in paths},
        control_bindings={str(p.relative_to(ROOT)): digest(p) for p in controls},
        held_outcomes_used=False, parameter_updates=0)


def call_remote(code, payload):
    handoff = HANDOFF/'compute_handoff_20260927.json'
    assert digest(handoff) == 'f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9'
    ssh = json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
    try:
        r = subprocess.run(ssh+[shlex.join(['/usr/bin/python3', '-c', code])],
            input=json.dumps(payload), capture_output=True, text=True, timeout=75)
        return dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except subprocess.TimeoutExpired:
        return dict(returncode=None, observation_timeout=True)


def submit(reg, replay=False, resume=False):
    label = 'replay' if replay else 'diagnose'
    path = PRIVATE/('submission_'+label+'.json')
    assert not path.exists(), 'Prior or uncertain submission: inspect, never duplicate'
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', str(PUBLIC/'registration.json')], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(['git', 'cat-file', '-e', 'HEAD:'+str((PUBLIC/'registration.json').relative_to(ROOT))], cwd=ROOT, check=True)
    flag = '--replay' if replay else '--resume' if resume else ''
    script = f'''#!/bin/bash -l
#SBATCH --job-name=m3w_easy_gradient_{label}
#SBATCH --partition=cpu
#SBATCH --account=kcl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output={label}-%j.out
#SBATCH --error={label}-%j.err
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
module load python/3.11.6-gcc-13.2.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset PYTHONPATH
../easy_hurdle_runtime_v2/venv/bin/python code/scripts/run_m3w_easy_gradient_diagnostic.py --home "$PWD" --parent-home {REMOTE_PARENT} {flag}
'''
    payload = dict(home=REMOTE, parent=REMOTE_PARENT, name=NAME, label=label,
        config=json.loads(CONFIG.read_text()), registration=reg,
        files={p: (ROOT/p).read_text() for p in reg['code_bindings']}, script=script)
    code = r'''
import hashlib,json,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['home']);parent=pathlib.Path(p['parent'])
assert json.loads((root.parent/'.m3w_owner.json').read_text())['project']=='M3W'
assert root.parent==parent.parent and root.name==p['name']
assert hashlib.sha256((parent/'input_manifest.json').read_bytes()).hexdigest()==p['config']['parent_manifest_sha256']
assert hashlib.sha256((parent/'training_complete.json').read_bytes()).hexdigest()==p['config']['parent_training_receipt_sha256']
queue=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=20)
assert queue.returncode==0 and 'm3w_easy_gradient_' not in queue.stdout
root.mkdir(exist_ok=True)
def write_once(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==text
    else:
        with path.open('x') as f:f.write(text)
write_once(root/'.owner.json',json.dumps({'project':'M3W','experiment':p['name']})+'\n')
intent=root/('submit_intent_'+p['label']+'.json');receipt=root/('submit_receipt_'+p['label']+'.json')
assert not intent.exists() and not receipt.exists(),'Prior or uncertain submission'
if p['label']=='replay':
    prior=json.loads((root/'receipt.json').read_text());assert prior['groups']==108
    q=subprocess.run(['sacct','-j',prior['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
    assert q.returncode==0 and q.stdout.strip()=='COMPLETED|0:0'
for name,value in [('config.json',p['config']),('registration.json',p['registration'])]:
    write_once(root/name,json.dumps(value,indent=2,allow_nan=False)+'\n')
for name,text in p['files'].items():
    assert name in p['registration']['code_bindings'] and not pathlib.PurePosixPath(name).is_absolute() and '..' not in pathlib.PurePosixPath(name).parts
    assert hashlib.sha256(text.encode()).hexdigest()==p['registration']['code_bindings'][name]
    write_once(root/'code'/name,text)
script=root/('run_'+p['label']+'.sh');write_once(script,p['script'])
write_once(intent,json.dumps({'script_sha256':hashlib.sha256(p['script'].encode()).hexdigest()})+'\n')
r=subprocess.run(['sbatch','--parsable',script.name],cwd=root,capture_output=True,text=True,timeout=30)
out=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
if r.returncode==0:
    out['job_id']=r.stdout.strip().split(';')[0];assert out['job_id'].isdigit()
write_once(receipt,json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''
    result = call_remote(code, payload)
    immutable(path, dict(utc=datetime.now(timezone.utc).isoformat(), label=label, result=result))
    print(json.dumps(dict(returncode=result['returncode'], receipt=str(path),
                         response=result.get('stdout', ''), uncertain=result['returncode'] is None)))


def inspect_or_collect(collect=False):
    code = r'''
import hashlib,json,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['home']);result={}
for label in ('diagnose','replay'):
    path=root/('submit_receipt_'+label+'.json')
    intent=root/('submit_intent_'+label+'.json')
    result[label+'_intent']=intent.exists()
    if path.exists():
        receipt=json.loads(path.read_text());result[label+'_submission']=receipt
        if receipt.get('job_id'):
            q=subprocess.run(['sacct','-j',receipt['job_id'],'--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed,MaxRSS'],capture_output=True,text=True,timeout=20)
            result[label+'_accounting']=dict(code=q.returncode,stdout=q.stdout)
for name in ('heartbeat.json','receipt.json','replay_receipt.json'):
    if (root/name).exists():result[name]=json.loads((root/name).read_text())
if p['collect']:
    for name in ('receipt.json','replay_receipt.json'):
        doc=result[name]
        q=subprocess.run(['sacct','-j',doc['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
        assert q.returncode==0 and q.stdout.strip()=='COMPLETED|0:0'
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
    if result['returncode'] != 0:
        print(json.dumps(dict(returncode=result['returncode'], observation='unavailable_not_terminal'))); return
    observed = json.loads(result['stdout'])
    if collect:
        assert len(observed['groups']) == 108
        immutable(PRIVATE/'collected.json', observed)
        print(json.dumps(dict(groups=108, replay_exact=True, local_record=str(PRIVATE/'collected.json'))))
    else:
        print(json.dumps(observed, indent=2)[:7000])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'submit', 'submit-replay', 'inspect', 'collect'])
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    reg = registration()
    if a.phase == 'register':
        immutable(PUBLIC/'registration.json', reg); print(json.dumps(dict(files=len(reg['code_bindings']), status='registered_before_execution')))
    else:
        assert json.loads((PUBLIC/'registration.json').read_text()) == reg
        if a.phase in ('submit', 'submit-replay'):
            submit(reg, replay=a.phase == 'submit-replay', resume=a.resume)
        else:
            inspect_or_collect(collect=a.phase == 'collect')


if __name__ == '__main__':
    main()
