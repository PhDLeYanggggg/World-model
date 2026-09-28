"""Register, submit and observe the single-factor repair without duplicate jobs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.manage_m3w_easy_gradient_diagnostic import call_remote, digest, immutable
from scripts.export_m3w_easy_hurdle_create import closure

NAME = 'european_easy_risk_priority_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_european_easy_risk_priority_v1.json'
DIAGNOSTIC = PUBLIC.parent/'european_easy_gradient_diagnostic_v1'
REMOTE = '/users/k24101830/m3w/'+NAME
PARENT = '/users/k24101830/m3w/european_easy_hurdle_v1'


def registration():
    cfg = json.loads(CONFIG.read_text())
    assert digest(DIAGNOSTIC/'verification.json') == cfg['diagnostic_seal_sha256']
    seal = json.loads((DIAGNOSTIC/'verification.json').read_text())
    for rel, sha in seal['source_bindings'].items():
        assert digest(ROOT/rel) == sha
    for rel, sha in seal['artifacts'].items():
        assert digest(DIAGNOSTIC/rel) == sha
    paths = closure(ROOT, ['scripts.train_m3w_easy_risk_priority_portable'])
    controls = [CONFIG, PUBLIC/'protocol.md', Path(__file__), ROOT/'tests/test_m3w_easy_risk_priority.py']
    return dict(name=NAME, diagnostic_seal_sha256=digest(DIAGNOSTIC/'verification.json'),
        code_bindings={str(p.relative_to(ROOT)): digest(p) for p in paths},
        control_bindings={str(p.relative_to(ROOT)): digest(p) for p in controls},
        arms=['uncapped', 'risk_priority'], primary_contrast=['risk_priority_matched', 'uncapped_matched'],
        groups=108, heads=216, updates_per_head=2000, held_outcomes_used=False,
        independent_roles_read=False, formal_primary_replaced=False)


def submit(phase, reg):
    record = PRIVATE/('submission_'+phase+'.json')
    assert not record.exists(), 'Inspect previous or uncertain submission; do not duplicate'
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', str(PUBLIC/'registration.json')], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(['git', 'cat-file', '-e', 'HEAD:'+str((PUBLIC/'registration.json').relative_to(ROOT))], cwd=ROOT, check=True)
    script = f'''#!/bin/bash -l
#SBATCH --job-name=m3w_risk_priority_{phase}
#SBATCH --partition=cpu
#SBATCH --account=kcl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output={phase}-%j.out
#SBATCH --error={phase}-%j.err
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
module load python/3.11.6-gcc-13.2.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset PYTHONPATH
../easy_hurdle_runtime_v2/venv/bin/python code/scripts/train_m3w_easy_risk_priority_portable.py --home "$PWD" --parent-home {PARENT} --phase {phase}
'''
    payload = dict(home=REMOTE, parent=PARENT, name=NAME, phase=phase,
        config=json.loads(CONFIG.read_text()), registration=reg, script=script,
        files={p: (ROOT/p).read_text() for p in reg['code_bindings']})
    code = r'''
import hashlib,json,os,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['home']);parent=pathlib.Path(p['parent'])
assert root.parent==parent.parent and root.name==p['name']
assert json.loads((root.parent/'.m3w_owner.json').read_text())['project']=='M3W'
assert hashlib.sha256((parent/'input_manifest.json').read_bytes()).hexdigest()==p['config']['parent_manifest_sha256']
assert hashlib.sha256((parent/'training_complete.json').read_bytes()).hexdigest()==p['config']['parent_training_receipt_sha256']
limit=int(os.getxattr('/users/k24101830','ceph.quota.max_bytes'));used=int(os.getxattr('/users/k24101830','ceph.dir.rbytes'))
assert limit>used+2*2**30,'Need verified personal-quota headroom; do not delete old data'
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and 'm3w_risk_priority_' not in q.stdout
root.mkdir(exist_ok=True)
def once(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==text
    else:
        with path.open('x') as f:f.write(text)
once(root/'.owner.json',json.dumps({'project':'M3W','experiment':p['name']})+'\n')
phase=p['phase'];intent=root/('submit_intent_'+phase+'.json');receipt=root/('submit_receipt_'+phase+'.json')
assert not intent.exists() and not receipt.exists(),'Prior submission must be inspected'
if phase!='pilot':
    previous=json.loads((root/('pilot.json' if phase=='train' else 'training_complete.json')).read_text())
    a=subprocess.run(['sacct','-j',previous['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
    assert a.returncode==0 and a.stdout.strip()=='COMPLETED|0:0'
    assert previous['groups']==(1 if phase=='train' else 108)
    if phase=='train':
        fit_seconds=previous['observed_fitting_seconds']
        projected=(fit_seconds*2000/p['config']['pilot_updates']+max(0,previous['seconds']-fit_seconds))*108
        assert projected<6800 and previous['peak_RSS_KiB']<12*2**20,'Revise resource envelope explicitly, not scientific scope'
for filename,value in [('config.json',p['config']),('registration.json',p['registration'])]:
    once(root/filename,json.dumps(value,indent=2,allow_nan=False)+'\n')
for name,text in p['files'].items():
    assert not pathlib.PurePosixPath(name).is_absolute() and '..' not in pathlib.PurePosixPath(name).parts
    assert hashlib.sha256(text.encode()).hexdigest()==p['registration']['code_bindings'][name]
    once(root/'code'/name,text)
script=root/('run_'+phase+'.sh');once(script,p['script'])
once(intent,json.dumps({'script_sha256':hashlib.sha256(p['script'].encode()).hexdigest(),'quota_limit_bytes':limit,'quota_used_bytes':used})+'\n')
r=subprocess.run(['sbatch','--parsable',script.name],cwd=root,capture_output=True,text=True,timeout=30)
out=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,quota_limit_bytes=limit,quota_used_bytes=used)
if r.returncode==0:
    out['job_id']=r.stdout.strip().split(';')[0];assert out['job_id'].isdigit()
once(receipt,json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''
    result = call_remote(code, payload)
    immutable(record, dict(utc=datetime.now(timezone.utc).isoformat(), phase=phase, result=result))
    print(json.dumps(dict(returncode=result['returncode'], record=str(record), response=result.get('stdout', ''), uncertain=result['returncode'] is None)))


def inspect(collect=False):
    code = r'''
import hashlib,json,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['home']);result={}
for phase in ('pilot','train','replay'):
    receipt=root/('submit_receipt_'+phase+'.json');result[phase+'_intent']=(root/('submit_intent_'+phase+'.json')).exists()
    if receipt.exists():
        doc=json.loads(receipt.read_text());result[phase+'_submission']=doc
        if doc.get('job_id'):
            a=subprocess.run(['sacct','-j',doc['job_id'],'--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed,MaxRSS'],capture_output=True,text=True,timeout=20)
            result[phase+'_accounting']=dict(returncode=a.returncode,stdout=a.stdout)
            error=root/(phase+'-'+doc['job_id']+'.err')
            if error.exists():result[phase+'_stderr_tail']=error.read_text()[-4000:]
for name in ('heartbeat.json','pilot.json','training_complete.json','replay.json'):
    if (root/name).exists():result[name]=json.loads((root/name).read_text())
if p['collect']:
    for name,count in [('training_complete.json',108),('replay.json',1)]:
        doc=result[name];assert doc['groups']==count
        a=subprocess.run(['sacct','-j',doc['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
        assert a.returncode==0 and a.stdout.strip()=='COMPLETED|0:0'
        assert doc['control_parent_states_exact']==count
        for ref in doc['artifacts']:
            assert hashlib.sha256((root/ref['path']).read_bytes()).hexdigest()==ref['sha256']
    assert result['replay.json']['repair_first_pair_replay_exact']
print(json.dumps(result))
'''
    result = call_remote(code, dict(home=REMOTE, collect=collect))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    immutable(PRIVATE/('observation_'+stamp+'.json'), result)
    if result['returncode'] != 0:
        print(json.dumps(dict(returncode=result['returncode'], observation='unavailable_not_terminal', stderr_tail=result.get('stderr','')[-600:]))); return
    observed = json.loads(result['stdout'])
    if collect:
        immutable(PRIVATE/'collected.json', observed)
    compact = {k: v for k, v in observed.items() if k not in ('pilot.json','training_complete.json','replay.json')}
    for k in ('pilot.json', 'training_complete.json', 'replay.json'):
        if k in observed:
            compact[k] = {a: b for a, b in observed[k].items() if a not in ('artifacts','fitting_summary')}
    print(json.dumps(compact, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register','pilot','train','replay','inspect','collect']); a = p.parse_args()
    reg = registration()
    if a.phase == 'register':
        immutable(PUBLIC/'registration.json', reg)
        print(json.dumps(dict(status='registered_before_training', source_files=len(reg['code_bindings']))))
    else:
        assert json.loads((PUBLIC/'registration.json').read_text()) == reg
        if a.phase in ('pilot','train','replay'):
            submit(a.phase, reg)
        else:
            inspect(collect=a.phase == 'collect')


if __name__ == '__main__':
    main()
