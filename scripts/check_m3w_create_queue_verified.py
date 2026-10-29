"""Read-only CREATE queue observation with explicit remote execution markers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import uuid

from scripts.check_m3w_create_queue_readonly import ROOT, SOURCE


def classify_response(response, nonce):
    begin=f'M3W_QUEUE_{nonce}_BEGIN'
    end=f'M3W_QUEUE_{nonce}_END:0'
    lines=response.get('stdout','').splitlines()
    verified=(response.get('returncode')==0 and lines.count(begin)==1 and
              lines.count(end)==1 and lines.index(begin)<lines.index(end))
    return dict(queue_snapshot_verified=verified,
        queue_lines=lines[lines.index(begin)+1:lines.index(end)] if verified else None,
        state='verified_queue_snapshot_only' if verified else 'unresolved_remote_execution_not_verified',
        job_completion_inferred=False,jobs_submitted=0,jobs_cancelled=0,remote_modified=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True);args=parser.parse_args()
    path=(ROOT/args.receipt).resolve()
    if not path.is_relative_to(ROOT/'data/stage_cvpr2027_experiments') or path.exists():
        raise ValueError('A new ignored private receipt is required')
    subprocess.run(['git','check-ignore','--quiet',str(path)],cwd=ROOT,check=True)
    original=json.loads(SOURCE.read_text());query=original['queries'][0];ssh=original['ssh_arguments']
    if (query['query']!='queue' or query['command'][0]!='squeue' or ssh[0]!='ssh'
            or 'BatchMode=yes' not in ssh or 'StrictHostKeyChecking=yes' not in ssh):
        raise ValueError('Use the existing authorized read-only SSH/squeue interface')
    nonce=uuid.uuid4().hex
    body=(f"printf '%s\\n' {shlex.quote('M3W_QUEUE_'+nonce+'_BEGIN')}; "
          f"{shlex.join(query['command'])}; status=$?; "
          f"printf '%s:%d\\n' {shlex.quote('M3W_QUEUE_'+nonce+'_END')} \"$status\"; exit \"$status\"")
    started=datetime.now(timezone.utc).isoformat()
    try:
        result=subprocess.run(ssh+[shlex.join(['sh','-c',body])],capture_output=True,text=True,timeout=60)
        response=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    except subprocess.TimeoutExpired:
        response=dict(returncode=None,observation_timeout=True)
    observation=classify_response(response,nonce)
    record=dict(started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),
        source_receipt_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        nonce=nonce,query='queue',response=response,observation=observation)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as out:json.dump(record,out,indent=2);out.write('\n')
    print(json.dumps(dict(returncode=response['returncode'],queue_snapshot_verified=observation['queue_snapshot_verified'],
        state=observation['state'],queue_line_count=len(observation['queue_lines']) if observation['queue_lines'] is not None else None,
        receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),jobs_submitted=0)))


if __name__=='__main__':main()
