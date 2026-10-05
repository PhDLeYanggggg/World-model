"""Add only missing pinned imports in an owned, allocated M3W environment."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

REQUIRED = {'pandas': '3.0.3', 'python-dateutil': '2.9.0.post0', 'six': '1.17.0'}
CORE = {'torch': '2.12.0+cpu', 'numpy': '2.4.6', 'scipy': '1.17.1', 'scikit-learn': '1.8.0'}


def inventory():
    return {d.metadata['Name'].lower().replace('_', '-'): d.version
            for d in importlib.metadata.distributions()}


def plan(before):
    if any(before.get(name) != value for name, value in CORE.items()):
        raise ValueError('Registered core runtime changed')
    if any(name in before and before[name] != value for name, value in REQUIRED.items()):
        raise ValueError('Existing dependency version must not be overwritten')
    return [name+'=='+value for name, value in REQUIRED.items() if name not in before]


def validate(before, after):
    if any(after.get(name) != value for name, value in before.items()):
        raise ValueError('An existing package changed')
    if set(after)-set(before) != set(REQUIRED)-set(before):
        raise ValueError('Unregistered package addition')
    if any(after.get(name) != value for name, value in REQUIRED.items()):
        raise ValueError('Pinned dependencies missing or mismatched')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args()
    job = os.environ.get('SLURM_JOB_ID'); assert job and job.isdigit()
    root, runtime = args.experiment.resolve(), args.runtime.resolve()
    parent = Path('/users/k24101830/m3w').resolve()
    assert root.parent == runtime.parent == parent
    assert root.name == 'european_temporal_auxiliary_v1' and runtime.name == 'easy_hurdle_runtime_v2'
    assert Path(sys.prefix).resolve() == (runtime/'venv').resolve()
    assert json.loads((root/'.owner.json').read_text())['experiment'] == root.name
    assert json.loads((runtime/'.m3w_runtime_owner.json').read_text())['project'] == 'M3W'
    registration = root/'dependency_repair_registration.json'
    intent = json.loads((root/'dependency_pilot_retry_intent.json').read_text())
    assert hashlib.sha256(registration.read_bytes()).hexdigest() == intent['registration_sha256']
    bindings = json.loads(registration.read_text())['bindings']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == bindings['scripts/repair_m3w_temporal_dependencies.py']
    with (runtime/'temporal_dependency_repair.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q = subprocess.run(['squeue', '--me', '-h', '-o', '%i|%j'], capture_output=True, text=True, timeout=20)
        assert q.returncode == 0
        assert all('|m3w_' not in line or line.split('|')[0] == job for line in q.stdout.splitlines())
        home = Path('/users/k24101830')
        free = int(os.getxattr(home, 'ceph.quota.max_bytes'))-int(os.getxattr(home, 'ceph.dir.rbytes'))
        assert free >= 10*2**30+256*2**20+2*2**20+512*2**20
        before = inventory(); packages = plan(before)
        record = root/'dependency_runtime_receipt.json'
        assert not record.exists(), 'Preserve and verify existing successful repair'
        if packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                '--no-deps', '--only-binary=:all:', '--no-cache-dir', '--index-url', 'https://pypi.org/simple',
                '--report', str(root/'dependency_install_report.json'), *packages], check=True, timeout=900)
        after = inventory(); validate(before, after)
        subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True, timeout=60)
        # Import the whole original entry point, not just Torch, before TRAIN access.
        subprocess.run([sys.executable, '-m', 'scripts.train_m3w_temporal_auxiliary_portable', '--help'],
                       cwd=root/'code', check=True, timeout=300)
        out = dict(result_source='fresh_run_allocated_runtime_dependency_repair', job_id=job,
            utc=datetime.now(timezone.utc).isoformat(), before=before, after=after, added=packages,
            existing_packages_unchanged=True, full_entry_import_passed=True, pip_check_passed=True,
            scientific_training_executed=False, numerical_code_changed=False, other_projects_touched=False)
        with record.open('x') as f: json.dump(out, f, indent=2); f.write('\n')
        print(json.dumps(out), flush=True)


if __name__ == '__main__': main()
