"""Bounded read-only observation of the owned fitting job and checkpoints."""
import argparse
from datetime import datetime, timezone
import json
import shlex
import subprocess
from scripts.prepare_m3w_create_runtime import PRIVATE, HANDOFF


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['pilot','train'],required=True);a=p.parse_args()
    s=json.loads((PRIVATE/('create_head_'+a.phase+'_submission.json')).read_text())
    r=json.loads(s['response']['stdout']);job=r['job_id'];assert job.isdigit()
    home=json.loads((PRIVATE/'remote_input_manifest.json').read_text())['remote_path']
    ssh=json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    code=r'''
import json,pathlib,subprocess,sys
home=pathlib.Path(sys.argv[1]);job=sys.argv[2];phase=sys.argv[3]
assert job.isdigit() and phase in ('pilot','train') and json.loads((home/'.owner.json').read_text())['experiment']=='european_easy_hurdle_v1'
r={}
for name,cmd in [('queue',['squeue','-j',job,'-h','-o','%i|%T|%M|%R']),('accounting',['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,Elapsed,ExitCode,MaxRSS'])]:
    q=subprocess.run(cmd,capture_output=True,text=True,timeout=20);r[name]={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
for name in ['heartbeat.json','pilot.json','training_complete.json',phase+'-'+job+'.out',phase+'-'+job+'.err']:
    p=home/name;r[name]=p.read_text()[-30000:] if p.is_file() else None
if (home/'training_complete.json').is_file():r['training_complete.json']=(home/'training_complete.json').read_text()
r['checkpoint_count']=len(list((home/'heads').glob('*/*/checkpoint.pt.gz')))
print(json.dumps(r))
'''
    try:
        q=subprocess.run(ssh+[shlex.join(['/usr/bin/python3','-c',code,home,job,a.phase])],capture_output=True,text=True,timeout=60)
        out=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
    except subprocess.TimeoutExpired:
        out=dict(returncode=None,observation_timeout=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');path=PRIVATE/f'create_head_{a.phase}_observation_{stamp}.json'
    with path.open('x') as f:
        f.write(json.dumps(dict(job_id=job,response=out,remote_modified=False),indent=2)+'\n')
    if out['returncode']==0:
        r=json.loads(out['stdout']);print(json.dumps(dict(job_id=job,queue=r['queue']['stdout'],accounting=r['accounting']['stdout'],
            heartbeat=json.loads(r['heartbeat.json']) if r['heartbeat.json'] else None,checkpoints=r['checkpoint_count'],
            training_receipt_present=r['training_complete.json'] is not None)))
    else:print(json.dumps(dict(job_id=job,observation_unavailable=True)))


if __name__=='__main__':
    main()
