"""Bounded same-allocation Slurm observations; no scientific execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

JOBS = ('37835856', '37835859', '37837636')
REMOTE = Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
REL = Path('outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1')
REG_SHA = 'a81c1dbbb08e410e6a40c97321752fc7ec1f8aeb81168fa0d81c7902762d99ef'
FREEZE_SHA = 'f57acbc74ba9afa47550b7c5aa37e70a42c2a14bd777ac9d4e9d8b2ee9542426'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(kind, explicit_cluster=True):
    if kind not in ('sacct', 'squeue'):
        raise ValueError('Only fixed owned-job observations are permitted')
    args = [kind]
    if explicit_cluster:
        args += ['--clusters=create']
    args += ['-j', ','.join(JOBS)]
    if kind == 'sacct':
        args += ['--starttime=2026-10-06', '--noheader', '--parsable2',
                 '--format=JobID,State,Elapsed,ExitCode,MaxRSS']
    else:
        args += ['-h', '-o', '%i|%T|%M|%R']
    return args


def query(kind, explicit_cluster=True):
    args = command(kind, explicit_cluster)
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=25)
        return dict(argv=args, returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return dict(argv=args, returncode=None, stdout='', stderr=str(exc),
                    observation='query_unavailable_status_unknown')


def collect(mode):
    if mode not in ('system', 'module'):
        raise ValueError('An explicit environment is required')
    job = os.environ.get('SLURM_JOB_ID', '')
    host = socket.gethostname()
    if not job.isdigit() or not os.environ.get('SLURM_JOB_NODELIST') or not host.startswith('erc-hpc-') or 'login' in host:
        raise RuntimeError('An allocated CREATE compute node is required')
    if mode == 'system' and (Path(sys.executable) != Path('/usr/bin/python3')
            or os.environ.get('LD_LIBRARY_PATH') or os.environ.get('LD_PRELOAD')):
        raise RuntimeError('System branch must use isolated system Python and loader environment')
    owner = json.loads((REMOTE / '.owner.json').read_text())
    home = REMOTE / REL
    if (owner['registration_sha256'] != REG_SHA or sha(REMOTE / 'registration.json') != REG_SHA
            or sha(home / 'training_freeze.json') != FREEZE_SHA):
        raise ValueError('Owned unchanged experiment required')
    # The experiment directory is new and exclusive, preventing an unnoticed resubmit.
    directory = home / 'web_accounting_environment_v2'
    if mode == 'system':
        directory.mkdir(exist_ok=False)
    else:
        first = json.loads((directory / ('system_' + job + '.json')).read_text())
        if first['job_id'] != job or first['node'] != host:
            raise ValueError('Same-allocation paired observation required')
    out = dict(schema='m3w_accounting_environment_v2', mode=mode, job_id=job, node=host,
        observed_utc=datetime.now(timezone.utc).isoformat(), worker_sha256=sha(__file__),
        registration_sha256=REG_SHA, freeze_sha256=FREEZE_SHA,
        python=dict(executable=sys.executable, version=sys.version.split()[0]),
        loader_environment_present={k:bool(os.environ.get(k)) for k in ('LD_LIBRARY_PATH', 'LD_PRELOAD')},
        accounting_explicit=query('sacct'), accounting_local=query('sacct', False),
        queue_local=query('squeue', False), existing_jobs_changed=False,
        checkpoint_bytes_read=0, scientific_execution=False, independent_roles_read=False,
        stage5c_executed=False, smc_enabled=False)
    path = directory / (mode + '_' + job + '.json')
    with path.open('x') as stream:
        json.dump(out, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(path), sha256=sha(path), mode=mode,
                          accounting_returncodes=[out[k]['returncode'] for k in
                              ('accounting_explicit', 'accounting_local')])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['system', 'module'])
    collect(parser.parse_args().mode)


if __name__ == '__main__':
    main()
