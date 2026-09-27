"""Create an isolated authorized M3W runtime trial and submit at most one job."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_easy_hurdle_v1'
HANDOFF = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--submit', action='store_true')
    p.add_argument('--revision',type=int,choices=[1,2],default=1); a = p.parse_args()
    suffix='' if a.revision==1 else '_v2'
    receipt = PRIVATE/(('create_runtime_submission' if a.submit else 'create_runtime_preparation')+suffix+'.json')
    if receipt.exists():
        raise ValueError('Inspect the existing operation; do not duplicate submission')
    subprocess.run(['git', 'check-ignore', '--quiet', str(receipt)], cwd=ROOT, check=True)
    hfile = HANDOFF/'compute_handoff_20260927.json'
    assert hashlib.sha256(hfile.read_bytes()).hexdigest() == 'f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9'
    h = json.loads(hfile.read_text()); project = h['paths']['possible_independent_new_project']
    protected = h['paths']['protected_simulation_root']
    assert project == h['paths']['user_home']+'/m3w' and protected not in project
    ssh = json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
    script='setup_m3w_create_cpu.sh' if a.revision==1 else 'setup_m3w_create_cpu_v2.sh'
    files = {'setup_m3w_create_cpu.sh':(ROOT/'scripts'/script).read_text(),
             'probe_m3w_create_torch.py':(ROOT/'scripts/probe_m3w_create_torch.py').read_text()}
    payload = dict(project=project, files=files, submit=a.submit,
                   revision=a.revision,
                   binding={k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()})
    remote = r'''
import hashlib, json, pathlib, subprocess, sys
p=json.loads(sys.stdin.read()); root=pathlib.Path(p['project']); marker=root/'.m3w_owner.json'
owner={'project':'M3W','purpose':'independent public-trajectory research','simulation_environment_used':False}
if root.exists():
    assert marker.is_file() and json.loads(marker.read_text())==owner, 'Existing directory lacks matching M3W owner marker'
else:
    root.mkdir(); marker.write_text(json.dumps(owner,sort_keys=True)+'\n')
assert p['revision'] in (1,2)
trial=root/('easy_hurdle_runtime_v'+str(p['revision'])); trial.mkdir(exist_ok=True)
m=trial/'.m3w_runtime_owner.json'; expected={'project':'M3W','binding':p['binding'],'research_training':False}
if m.exists(): assert json.loads(m.read_text())==expected
else: m.write_text(json.dumps(expected,sort_keys=True)+'\n')
for name,content in p['files'].items():
    path=trial/name
    if path.exists(): assert path.read_text()==content
    else: path.write_text(content)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==p['binding'][name]
result={'prepared':True,'trial_path':str(trial),'bindings':p['binding'],'jobs_submitted':0}
if p['submit']:
    intent=trial/'submission_intent.json'; done=trial/'submission_receipt.json'
    assert not intent.exists() and not done.exists(), 'Prior submit or uncertain attempt: inspect scheduler, never duplicate'
    q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%P|%T'],capture_output=True,text=True,timeout=30)
    assert q.returncode==0 and 'm3w_easy_cpu_runtime_' not in q.stdout, 'Queue query failed or runtime job already exists'
    intent.write_text(json.dumps({'binding':p['binding'],'queue_before':q.stdout})+'\n')
    r=subprocess.run(['sbatch','--parsable','setup_m3w_create_cpu.sh'],cwd=trial,capture_output=True,text=True,timeout=30)
    result.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,queue_before=q.stdout)
    if r.returncode==0:
        job=r.stdout.strip().split(';')[0]; assert job.isdigit(); result.update(job_id=job,jobs_submitted=1)
    done.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''
    start = datetime.now(timezone.utc).isoformat()
    try:
        r = subprocess.run(ssh+[shlex.join(['/usr/bin/python3', '-c', remote])], input=json.dumps(payload),
                           text=True, capture_output=True, timeout=60)
        out = dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except subprocess.TimeoutExpired:
        out = dict(returncode=None, observation_timeout=True, retry_requires_remote_intent_inspection=True)
    record = dict(started_utc=start, completed_utc=datetime.now(timezone.utc).isoformat(), operation='submit' if a.submit else 'prepare',
                  response=out, source_hashes=payload['binding'], simulation_touched=False, credentials_changed=False)
    with receipt.open('x') as f:
        f.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(returncode=out['returncode'], receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                         operation=record['operation'])))


if __name__ == '__main__':
    main()
