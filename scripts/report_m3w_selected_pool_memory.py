"""Resource-only in-memory execution of the original frozen accounting report."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_m3w_selected_pool_create as collector
from scripts import plot_m3w_selected_pool_source as source
from scripts import report_m3w_selected_pool_findings as findings

run = collector.run
PUBLIC = run.PUBLIC
CAP = 64 * 2**20


class MemoryPath:
    """Only the flat-file methods used by the frozen report; no disk fallback."""
    def __init__(self, files, name='', cap=CAP):
        self.files, self.name, self.cap = files, name, cap

    def __truediv__(self, name):
        if self.name or not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Only one bounded artifact directory is supported')
        return MemoryPath(self.files, name, self.cap)

    def exists(self):
        return self.name in self.files

    def is_file(self):
        return self.exists()

    def read_bytes(self):
        return self.files[self.name]

    def read_text(self):
        return self.read_bytes().decode()

    def write_text(self, text):
        data = text.encode()
        needed = sum(map(len, self.files.values())) - len(self.files.get(self.name, b'')) + len(data)
        if not self.name or needed > self.cap:
            raise ValueError('Memory artifact cap exceeded')
        self.files[self.name] = data
        return len(text)

    def iterdir(self):
        if self.name:
            raise ValueError('Not a directory')
        return [self / name for name in self.files]


def digest(path, algorithm='sha256'):
    if isinstance(path, MemoryPath):
        return hashlib.new(algorithm, path.read_bytes()).hexdigest()
    return run.digest(path, algorithm)


def immutable(path, value):
    if not isinstance(path, MemoryPath):
        raise ValueError('Numeric output must stay in memory')
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Existing immutable memory artifact differs')
    else:
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def execute_original_report(rows, cfg, reg, public_files, *, render_findings=True):
    assert len(rows) == 1080
    assert len({(r['role'], r['view'], r['mode']) for r in rows}) == 1080
    for role, expected in (('source_oof', 72), ('source_resubstitution', 72), ('transfer', 216)):
        for mode in run.parent.api.MODES:
            assert sum(r['role'] == role and r['mode'] == mode for r in rows) == expected
    files = dict(public_files)
    assert sum(map(len, files.values())) < CAP
    home = MemoryPath(files)
    namespace = dict(run.report.__globals__, PUBLIC=home, read_rows=lambda: rows,
        immutable=immutable, digest=digest)
    # Run the frozen function's actual bytecode, substituting storage only.
    FunctionType(run.report.__code__, namespace)(cfg, reg)
    if render_findings:
        proxy = SimpleNamespace(**{k: namespace[k] for k in ('PUBLIC', 'read_rows', 'immutable', 'digest', 'parent')})
        FunctionType(findings.main.__code__, dict(findings.main.__globals__, run=proxy))()
    return files


def registration():
    cfg, original = run.registration()
    assert json.loads((PUBLIC/'registration.json').read_text()) == original
    paths = [Path(__file__), ROOT/'scripts/collect_m3w_selected_pool_create.py',
        ROOT/'scripts/report_m3w_selected_pool_findings.py', ROOT/'tests/test_m3w_selected_pool_memory.py',
        PUBLIC/'memory_report_amendment.md']
    return dict(original_registration_sha256=run.digest(PUBLIC/'registration.json'),
        recovery_registration_sha256=run.digest(PUBLIC/'recovery_registration.json'),
        code_and_protocol={str(p.relative_to(ROOT)): run.digest(p) for p in paths},
        memory_cap_bytes=CAP, disk_reserve_bytes_unchanged=cfg['disk_reserve_bytes'],
        risk_budget=cfg['risk_budget'], bootstrap_resamples=cfg['bootstrap_resamples'],
        bootstrap_seed=cfg['bootstrap_seed'], new_training=False, policy_changed=False,
        independent_roles_read=False, jobs_submitted=0)


def store(files):
    from scripts import manage_m3w_selected_pool_create as manager
    code = r'''
import base64,hashlib,json,pathlib,sys
p=json.loads(sys.stdin.read());root=pathlib.Path('/users/k24101830/m3w/european_selected_pool_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_selected_pool_v1'
target=root/'memory_report_v1';target.mkdir(exist_ok=True);total=0
assert len(p['files'])<=100
for name,entry in p['files'].items():
    assert pathlib.PurePath(name).name==name and name not in ('.','..')
    raw=base64.b64decode(entry['base64']);total+=len(raw);assert total<=64*2**20
    assert hashlib.sha256(raw).hexdigest()==entry['sha256']
    f=target/name
    if f.exists():assert f.read_bytes()==raw
    else:
        with f.open('xb') as stream:stream.write(raw)
    assert hashlib.sha256(f.read_bytes()).hexdigest()==entry['sha256']
print(json.dumps(dict(verified_files=len(p['files']),bytes=total)))
'''
    return manager.remote(code, dict(files={k: dict(base64=base64.b64encode(v).decode(),
        sha256=hashlib.sha256(v).hexdigest()) for k, v in files.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'report'))
    args = parser.parse_args()
    registered = registration()
    path = PUBLIC/'memory_report_registration.json'
    if args.phase == 'register':
        run.immutable(path, registered)
        print('Resource-only memory report registered; no remote job submitted')
        return
    assert json.loads(path.read_text()) == registered
    run.base.inter.committed(path)
    assert platform.system() != 'Darwin' or platform.machine() == 'arm64'
    cfg, original = run.registration()
    bundle, manifest, job, total = collector.fetch_verified_bundle()
    source_rows, _ = source.load_source_rows()
    rows = list(source_rows); existing = 0
    for ref in bundle['complete']['groups']:
        text = bundle['files'][ref['group']]
        present, _ = collector.verify_existing(run.PRIVATE/'transfer'/(ref['group']+'.json'), text)
        existing += int(present)
        rows.extend(json.loads(text)['rows'])
    assert existing >= manifest['local_parity_groups']
    public_files = {p.name: p.read_bytes() for p in PUBLIC.iterdir() if p.is_file()}
    memory = MemoryPath(public_files)
    immutable(memory/'transfer_replay.json', dict(all216_exact_replay=True,
        method='two allocated-node complete passes plus exact parsed-value local parity', job_id=job))
    immutable(memory/'create_complete.json', bundle['complete'])
    started = time.monotonic()
    files = execute_original_report(rows, cfg, original, public_files)
    # Repeat the same numerical reduction; timings/test text are not numerical evidence.
    replay = execute_original_report(rows, cfg, original, public_files, render_findings=False)
    for name in ('readout.json', 'summary.json'):
        assert files[name] == replay[name], 'In-memory complete summary replay must be byte-exact'
    outputs = {k: files[k] for k in ('readout.json', 'summary.json', 'verification.json', 'scoped_tests.txt',
        'results.md', 'case_details.json', 'findings_verification.json', 'transfer_replay.json', 'create_complete.json')}
    receipt = dict(result_source='fresh_full_summary_on_cached_verified_complete_accounting',
        job_id=job, remote_accounting=bundle['accounting'], transfer_views=216, source_heads=72,
        existing_local_groups_exact=existing, parent_risk_checks=2592, remote_output_bytes=total,
        summary_exact_replay=True, seconds=time.monotonic()-started, local_numeric_artifacts_written=False,
        computation='local_native_RAM_original_frozen_report_function; CREATE_file_storage_only',
        registration_sha256=run.digest(path), original_registration_sha256=registered['original_registration_sha256'],
        disk_reserve_bytes_unchanged=cfg['disk_reserve_bytes'], policy_changed=False, new_training=False,
        source_public_artifacts={name: hashlib.sha256(data).hexdigest() for name, data in public_files.items()},
        outputs={name: dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in outputs.items()})
    outputs['memory_report_receipt.json'] = (json.dumps(receipt, indent=2)+'\n').encode()
    outputs['memory_report_registration.json'] = path.read_bytes()
    stored = store(outputs)
    print(json.dumps(dict(receipt=receipt, remote_storage=stored,
        screened_joint=json.loads(outputs['findings_verification.json'])['source_screened_joint_counts'])))


if __name__ == '__main__':
    main()
