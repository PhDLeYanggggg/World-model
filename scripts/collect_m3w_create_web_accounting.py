"""Collect owned scheduler evidence on CREATE compute through Open OnDemand."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

REMOTE = Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
REL = Path('outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1')
REG_SHA = 'a81c1dbbb08e410e6a40c97321752fc7ec1f8aeb81168fa0d81c7902762d99ef'
FREEZE_SHA = 'f57acbc74ba9afa47550b7c5aa37e70a42c2a14bd777ac9d4e9d8b2ee9542426'
JOBS = {'train': '37835856', 'join': '37835859', 'portability': '37837636'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query(kind, job):
    if kind not in ('squeue', 'sacct') or job not in JOBS.values():
        raise ValueError('Only owned read-only job queries allowed')
    command = [kind, '--clusters=create', '-j', job]
    if kind == 'squeue':
        command += ['-h', '-o', '%i|%T|%M|%R']
    else:
        command += ['--starttime=2026-10-06', '--noheader', '--parsable2',
                    '--format=JobID,State,Elapsed,ExitCode,MaxRSS']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=35)
        return dict(argv=command, returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired:
        return dict(argv=command, returncode=None, stdout='', stderr='',
                    observation='timeout_status_unknown')


def successful_accounting(record, job):
    if record.get('returncode') != 0:
        return False
    rows = [line.split('|') for line in record['stdout'].splitlines()]
    return any(len(row) >= 4 and row[0] == job and row[1] == 'COMPLETED'
               and row[3] == '0:0' for row in rows)


def collect():
    job = os.environ.get('SLURM_JOB_ID', '')
    host = socket.gethostname()
    if (not job.isdigit() or not os.environ.get('SLURM_JOB_NODELIST')
            or not host.startswith('erc-hpc-') or 'login' in host):
        raise RuntimeError('An allocated CREATE compute node is required')
    owner = json.loads((REMOTE / '.owner.json').read_text())
    if owner['registration_sha256'] != REG_SHA or sha(REMOTE / 'registration.json') != REG_SHA:
        raise ValueError('Owned frozen experiment required')
    home = REMOTE / REL
    if sha(home / 'training_freeze.json') != FREEZE_SHA:
        raise ValueError('Recovered freeze differs')
    output = home / 'web_accounting_v1'
    output.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    collection = {}
    for kind, owned in JOBS.items():
        name = 'node_portability_v1' if kind == 'portability' else kind
        submission = json.loads((REMOTE / (name + '_submission.json')).read_text())
        if submission['job_id'] != owned:
            raise ValueError('Submission ownership changed')
        collection[kind] = dict(submission=submission, queue=query('squeue', owned),
                                accounting=query('sacct', owned))
    collection['artifact'] = json.loads((home / 'training_freeze.json').read_text())
    files = {'training_freeze.json': (home / 'training_freeze.json').read_text()}
    for ref in collection['artifact']['fits']:
        path = (REMOTE / ref['path']).resolve()
        if not path.is_relative_to((home / 'fits').resolve()) or path.stat().st_size > 2 * 2**20:
            raise ValueError('Only lightweight owned fit receipts allowed')
        if sha(path) != ref['sha256']:
            raise ValueError('Fit receipt changed')
        files[str(path.relative_to(home.resolve()))] = path.read_text()
    if len(files) != 145 or sum(len(raw.encode()) for raw in files.values()) > 8 * 2**20:
        raise ValueError('Exact bounded metadata inventory required')
    report = dict(schema='m3w_web_accounting_v1', job_id=job, node=host,
        worker_sha256=sha(__file__), started_utc=started,
        observed_utc=datetime.now(timezone.utc).isoformat(),
        source_registration_sha256=REG_SHA, freeze_sha256=FREEZE_SHA,
        collection=collection, files=files, observations_sequential_not_atomic=True,
        existing_jobs_changed=False, checkpoint_bytes_read=0, scientific_execution=False,
        independent_roles_read=False, stage5c_executed=False, smc_enabled=False)
    path = output / ('accounting_' + job + '.json')
    with path.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    complete = successful_accounting(collection['join']['accounting'], JOBS['join'])
    print(json.dumps(dict(output=str(path), sha256=sha(path), join_accounting_pass=complete,
                          portability_accounting_pass=successful_accounting(
                              collection['portability']['accounting'], JOBS['portability']))), flush=True)
    return 0 if complete else 1


def accept(path):
    from scripts import read_m3w_easy_harm_final_heads_v2 as heads
    raw = path.read_bytes()
    if len(raw) > 8 * 2**20:
        raise ValueError('Oversized accounting receipt')
    report = json.loads(raw)
    if (report['schema'] != 'm3w_web_accounting_v1' or report['worker_sha256'] != sha(__file__)
            or report['source_registration_sha256'] != REG_SHA or report['freeze_sha256'] != FREEZE_SHA
            or any(report[k] is not False for k in ('existing_jobs_changed', 'scientific_execution',
                       'independent_roles_read', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Unchanged read-only worker evidence required')
    collection = report['collection']
    if not successful_accounting(collection['join']['accounting'], JOBS['join']):
        raise ValueError('Authoritative successful join accounting required')
    home = heads.manager.PUBLIC
    if json.loads((home / 'training_freeze.json').read_text()) != collection['artifact']:
        raise ValueError('Previously collected freeze changed')
    expected = {'training_freeze.json', *('fits/' + Path(ref['path']).name
                                       for ref in collection['artifact']['fits'])}
    if set(report['files']) != expected:
        raise ValueError('Exact collected metadata set required')
    for name, value in report['files'].items():
        if (home / name).read_bytes() != value.encode():
            raise ValueError('Web and compute receipt bytes disagree')
    for kind, job in JOBS.items():
        if collection[kind]['submission']['job_id'] != job:
            raise ValueError('Owned scheduler identities required')
    heads.manager.base.once(home / 'train_collection.json', collection)
    docs, _ = heads.local_admission()
    receipt = dict(result_source='fresh_run_CREATE_scheduler_and_artifact_collection',
        job_id=report['job_id'], observed_utc=report['observed_utc'], report_sha256=sha(path),
        worker_sha256=report['worker_sha256'], unique_fits=len(docs), scheduler_acceptance=True,
        scientific_readout='not_run', independent_roles_read=False, checkpoint_bytes_read=0,
        stage5c_executed=False, smc_enabled=False,
        accounting={kind: collection[kind]['accounting'] for kind in JOBS})
    heads.manager.base.once(home / 'web_accounting_v1' / 'accepted.json', receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    commands.add_parser('collect')
    sub = commands.add_parser('accept')
    sub.add_argument('report', type=Path)
    args = parser.parse_args()
    if args.mode == 'collect':
        return collect()
    accept(args.report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
