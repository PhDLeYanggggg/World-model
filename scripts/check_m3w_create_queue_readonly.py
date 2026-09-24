"""One bounded queue check using the existing approved CREATE handoff."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/stage_cvpr2027_experiments/create_handoff_20260923/observations.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    path = (ROOT / args.receipt).resolve()
    if not path.is_relative_to(ROOT / 'data/stage_cvpr2027_experiments') or path.exists():
        raise ValueError('A new private receipt is required')
    subprocess.run(['git', 'check-ignore', '--quiet', str(path)], cwd=ROOT, check=True)
    old = json.loads(SOURCE.read_text())
    query = old['queries'][0]
    assert query['query'] == 'queue' and query['command'][0] == 'squeue'
    ssh = old['ssh_arguments']
    assert ssh[0] == 'ssh' and 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
    start = datetime.now(timezone.utc).isoformat()
    try:
        result = subprocess.run(ssh + [shlex.join(query['command'])], capture_output=True, text=True, timeout=60)
        response = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired:
        response = dict(returncode=None, observation_timeout=True)
    receipt = dict(source_receipt_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                   started_utc=start, completed_utc=datetime.now(timezone.utc).isoformat(),
                   query='queue', response=response, jobs_submitted=0, remote_modified=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        out.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(returncode=response['returncode'],
                          receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          remote_job_state_known=response['returncode'] == 0)))


if __name__ == '__main__':
    main()
