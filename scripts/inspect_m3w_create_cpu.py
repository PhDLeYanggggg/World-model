"""Bounded read-only CPU/runtime/quota inspection; no remote writes or jobs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'
DEST = ROOT/'data/stage_cvpr2027_experiments/european_easy_hurdle_v1/create_cpu_inspection.json'


def main():
    if DEST.exists():
        raise ValueError('Do not overwrite a remote observation receipt')
    subprocess.run(['git', 'check-ignore', '--quiet', str(DEST)], cwd=ROOT, check=True)
    handoff = HOME/'compute_handoff_20260927.json'
    assert hashlib.sha256(handoff.read_bytes()).hexdigest() == 'f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9'
    info = json.loads(handoff.read_text())
    ssh = json.loads((HOME/'observations.json').read_text())['ssh_arguments']
    assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
    root = info['paths']['user_home']
    commands = [
        ['bash', '-lc', 'for p in python3.11 python3.12 python3.13 uv quota getfattr; do command -v "$p" || true; done'],
        ['bash', '-lc', 'module -t avail Python 2>&1'],
        ['bash', '-lc', 'if command -v quota >/dev/null; then quota -s; else printf "quota_command_unavailable\\n"; fi'],
        ['bash', '-lc', 'if command -v getfattr >/dev/null; then getfattr -n ceph.quota.max_bytes '+shlex.quote(root)+'; else printf "quota_xattr_tool_unavailable\\n"; fi'],
        ['df', '-Pk', root],
    ]
    results = []
    for command in commands:
        try:
            p = subprocess.run(ssh+[shlex.join(command)], text=True, capture_output=True, timeout=60)
            result = dict(command=command, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except subprocess.TimeoutExpired:
            result = dict(command=command, returncode=None, observation_timeout=True)
        results.append(result)
    receipt = dict(observed_utc=datetime.now(timezone.utc).isoformat(), handoff_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),
                   observations=results, remote_modified=False, jobs_submitted=0, simulation_touched=False)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open('x') as out:
        out.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(receipt_sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
        returncodes=[r['returncode'] for r in results], remote_modified=False)))


if __name__ == '__main__':
    main()
