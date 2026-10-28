"""Allocated, read-only verification of the complete frozen final-head inventory."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

REMOTE = Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
DIRECTORY = 'readout_artifact_v3'
CAP = 256 * 2**20


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def verify_files(root, manifest):
    expected = manifest['files']
    if (len(expected) != 289 or manifest['checkpoint_count'] != 144
            or sum(r['kind'] == 'checkpoint' for r in expected) != 144
            or sum(r['kind'] == 'receipt' for r in expected) != 144
            or sum(r['kind'] == 'freeze' for r in expected) != 1):
        raise ValueError('Exactly 144 checkpoints, 144 receipts and one freeze required')
    checked = {}
    total = 0
    for ref in expected:
        rel = PurePosixPath(ref['path'])
        prefix = ('data/stage_cvpr2027_experiments/european_easy_harm_deviance_v1/heads'
                  if ref['kind'] == 'checkpoint' else
                  'outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1')
        path = root / rel
        if (rel.is_absolute() or '..' in rel.parts or str(rel) != ref['path']
                or not rel.is_relative_to(prefix) or ref['path'] in checked
                or path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or not 0 < ref['bytes'] <= 2 * 2**20
                or path.stat().st_size != ref['bytes'] or sha(path) != ref['sha256']):
            raise ValueError('Changed, oversized, duplicated or misplaced frozen artifact')
        total += ref['bytes']
        if total > CAP:
            raise ValueError('Bounded inventory exceeded')
        checked[ref['path']] = dict(sha256=ref['sha256'], bytes=ref['bytes'], kind=ref['kind'])
    return checked, total


def main():
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or not os.environ.get('SLURM_JOB_NODELIST'):
        raise RuntimeError('Scheduler allocation required')
    root = REMOTE.resolve()
    work = root / DIRECTORY
    manifest = json.loads((work / 'manifest.json').read_text())
    if sha(__file__) != manifest['worker_sha256']:
        raise ValueError('Worker identity changed')
    sys.path.insert(0, str(root / 'code'))
    from scripts.train_m3w_easy_harm_recovery_v2 import verify
    _, amendment, _ = verify(root)
    if (sha(root / 'registration.json') != manifest['training_registration_sha256']
            or sha(root / 'control_execution_amendment_v2.json') != manifest['amendment_sha256']
            or len(amendment['preserved_fit_refs']) != 110):
        raise ValueError('Original training, references and preservation required')
    checked, total = verify_files(root, manifest)
    report = dict(schema='m3w_frozen_artifact_verification_v3', job_id=job,
        observed_utc=datetime.now(timezone.utc).isoformat(),
        slurm_nodelist=os.environ['SLURM_JOB_NODELIST'],
        worker_sha256=sha(__file__), manifest_sha256=sha(work / 'manifest.json'),
        files=checked, bytes_verified=total, original_training_bindings_verified=True,
        historical_scheduler_status='unavailable_not_inferred',
        scientific_execution=False, model_deserialization=False, new_training=False,
        existing_artifacts_changed=False, independent_roles_read=False,
        stage5c_executed=False, smc_enabled=False)
    with (work / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(state='all144_payloads_verified', job_id=job,
                         report_sha256=sha(work / 'report.json'), bytes_verified=total)), flush=True)


if __name__ == '__main__':
    main()
