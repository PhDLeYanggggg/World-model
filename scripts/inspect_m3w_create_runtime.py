"""Read one owned runtime job and its output without changing scheduler state."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from scripts.prepare_m3w_create_runtime import ROOT, PRIVATE, HANDOFF


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--revision',type=int,choices=[1,2],default=1)
    a=parser.parse_args();suffix='' if a.revision==1 else '_v2'
    submitted = json.loads((PRIVATE/('create_runtime_submission'+suffix+'.json')).read_text())
    result = json.loads(submitted['response']['stdout']); job = result['job_id']
    assert result['jobs_submitted'] == 1 and job.isdigit()
    trial = result['trial_path']
    ssh = json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    code = r'''
import json,pathlib,subprocess,sys
trial=pathlib.Path(sys.argv[1]); job=sys.argv[2]; assert job.isdigit()
assert json.loads((trial/'.m3w_runtime_owner.json').read_text())['project']=='M3W'
r={}
for name,cmd in [('queue',['squeue','-j',job,'-h','-o','%i|%T|%M|%R']),('accounting',['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,Elapsed,ExitCode,MaxRSS'])]:
    q=subprocess.run(cmd,capture_output=True,text=True,timeout=20); r[name]={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
for name in ['runtime_receipt.json','environment-freeze.txt','runtime-'+job+'.out','runtime-'+job+'.err']:
    p=trial/name; r[name]=p.read_text()[-16000:] if p.is_file() else None
print(json.dumps(r))
'''
    start = datetime.now(timezone.utc).isoformat()
    try:
        q = subprocess.run(ssh+[shlex.join(['/usr/bin/python3', '-c', code, trial, job])],
                           capture_output=True, text=True, timeout=60)
        out = dict(returncode=q.returncode, stdout=q.stdout, stderr=q.stderr)
    except subprocess.TimeoutExpired:
        out = dict(returncode=None, observation_timeout=True)
    name = datetime.now(timezone.utc).strftime('create_runtime_observation'+suffix+'_%Y%m%dT%H%M%SZ.json')
    record = dict(started_utc=start, job_id=job, response=out, remote_modified=False)
    with (PRIVATE/name).open('x') as f:
        f.write(json.dumps(record, indent=2)+'\n')
    if out['returncode'] == 0:
        r = json.loads(out['stdout'])
        print(json.dumps(dict(job_id=job, queue=r['queue']['stdout'].strip(),
            accounting=r['accounting']['stdout'].strip(), runtime_receipt_present=r['runtime_receipt.json'] is not None,
            receipt=name)))
    else:
        print(json.dumps(dict(job_id=job, observation_unavailable=True, receipt=name)))


if __name__ == '__main__':
    main()
