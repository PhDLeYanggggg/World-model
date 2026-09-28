"""Restore exact owned CREATE checkpoints after completed training verification."""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.manage_m3w_easy_risk_priority import PUBLIC, PRIVATE, REMOTE, registration, digest, immutable
from scripts.manage_m3w_easy_gradient_diagnostic import HANDOFF
from src.world_model import m3w_easy_risk_priority as repair

RESERVE = 10*2**30


def install_bundle(content, refs, destination, reserve):
    expected = {r['path']: r['sha256'] for r in refs}
    if len(expected) != len(refs):
        raise ValueError('Duplicate checkpoint reference')
    with tarfile.open(fileobj=io.BytesIO(content), mode='r:') as tar:
        members = tar.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('Exact frozen checkpoint set required')
        values = []
        for m in members:
            rel = PurePosixPath(m.name)
            if (not m.isfile() or rel.is_absolute() or '..' in rel.parts or len(rel.parts) != 4
                    or rel.parts[0] != 'heads' or rel.parts[2] not in repair.ARMS
                    or rel.parts[3] != 'checkpoint.pt.gz'):
                raise ValueError('Unsafe or unrelated archive path')
            data = tar.extractfile(m).read()
            if hashlib.sha256(data).hexdigest() != expected[m.name]:
                raise ValueError('Checkpoint checksum mismatch')
            path = destination/m.name
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Checkpoint destination escapes through a symlink')
            values.append((path, data))
        # Validate every member before any installation; retain matching files on resume.
        for path, data in values:
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError('Existing checkpoint differs; no overwrite')
                continue
            reserve(len(data)); path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.restore-', delete=False) as f:
                temp = Path(f.name)
                try:
                    f.write(data); f.flush(); os.fsync(f.fileno())
                except BaseException:
                    temp.unlink(missing_ok=True); raise
            os.replace(temp, path)
    return sum(len(d) for _, d in values)


def main():
    reg = registration()
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    from scripts.run_m3w_easy_risk_priority_policy import identity as decision_identity
    assert decision_identity() == json.loads((PUBLIC/'decision_registration.json').read_text())
    from scripts import run_m3w_easy_hurdle as source
    repair.torch.set_num_threads(4); repair.torch.set_num_interop_threads(1)
    for path in (PUBLIC/'create_training_freeze.json', PUBLIC/'decision_registration.json'):
        source.base.inter.committed(path)
    frozen = json.loads((PUBLIC/'create_training_freeze.json').read_text())
    first, replay = frozen['receipt'], frozen['replay_receipt']
    assert first['registration_sha256'] == replay['registration_sha256'] == digest(PUBLIC/'registration.json')
    assert first['groups'] == first['control_parent_states_exact'] == 108
    assert first['heads'] == len(first['artifacts']) == 216 and replay['repair_first_pair_replay_exact']
    payload = dict(home=REMOTE, registration_sha256=digest(PUBLIC/'registration.json'),
                   receipt=first, replay_receipt=replay)
    if shutil.disk_usage(PRIVATE).free < RESERVE+800_000_000:
        raise OSError('Keep10GiB plus800MB checkpoint/action allowance; do not delete old files')
    code = r'''
import hashlib,io,json,pathlib,subprocess,sys,tarfile
p=json.loads(sys.stdin.read());root=pathlib.Path(p['home'])
assert json.loads((root/'.owner.json').read_text())=={'project':'M3W','experiment':'european_easy_risk_priority_v1'}
assert hashlib.sha256((root/'registration.json').read_bytes()).hexdigest()==p['registration_sha256']
for file,key in [('training_complete.json','receipt'),('replay.json','replay_receipt')]:
    doc=json.loads((root/file).read_text());assert doc==p[key]
    a=subprocess.run(['sacct','-j',doc['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
    assert a.returncode==0 and a.stdout.strip()=='COMPLETED|0:0'
refs=p['receipt']['artifacts'];assert len(refs)==216
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
    for ref in refs:
        rel=pathlib.PurePosixPath(ref['path'])
        assert rel==pathlib.PurePosixPath('heads')/ref['group']/ref['arm']/'checkpoint.pt.gz'
        assert not rel.is_absolute() and '..' not in rel.parts and ref['arm'] in ('uncapped','risk_priority')
        data=(root/rel).read_bytes();assert hashlib.sha256(data).hexdigest()==ref['sha256']
        entry=tarfile.TarInfo(str(rel));entry.size=len(data);entry.mode=0o600
        tar.addfile(entry,io.BytesIO(data))
'''
    ssh = json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    assert 'StrictHostKeyChecking=yes' in ssh and 'BatchMode=yes' in ssh
    def reserve(needed):
        if shutil.disk_usage(PRIVATE).free-needed < RESERVE:
            raise OSError('Preserve10GiB and completed checkpoints')
    with (PRIVATE/'restore.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = subprocess.run(ssh+[shlex.join(['/usr/bin/python3','-c',code])],
                           input=json.dumps(payload).encode(), capture_output=True, timeout=180)
        if r.returncode:
            raise RuntimeError(r.stderr.decode(errors='replace')[-1500:])
        size = install_bundle(r.stdout, first['artifacts'], PRIVATE, reserve)
        grouped = {}
        for ref in first['artifacts']:
            grouped.setdefault(ref['group'], {})[ref['arm']] = ref
        records = []
        for name, refs in grouped.items():
            assert set(refs) == set(repair.ARMS)
            states = {arm: repair.api.head.read_checkpoint(PRIVATE/ref['path']) for arm, ref in refs.items()}
            repair.assert_matched(states['uncapped'], states['risk_priority'])
            assert states['uncapped']['identity'] == states['risk_priority']['identity']
            ident = states['uncapped']['identity']
            assert ident['registration_sha256'] == digest(PUBLIC/'registration.json')
            assert all(s['step'] == 2000 for s in states.values())
            roles = ident['source_identity']['roles']
            source.base.floor_api.api.assert_roles(roles['producer_sites'], roles['controller_sites'], roles['training_sites'], roles['held_sites'])
            old = repair.api.head.read_checkpoint(source.PRIVATE/'heads'/name/'supervised'/'checkpoint.pt.gz')
            repair.assert_parent_control(old, states['uncapped'])
            assert old['identity'] == ident['source_identity']
            artifacts = {arm: source.base.artifact(PRIVATE/ref['path']) for arm, ref in refs.items()}
            record = dict(group=name, identity=ident, artifacts=artifacts, paired_initialization_and_samples=True,
                          held_outcomes_used=False, parent_control_state_exact=True)
            path = PRIVATE/'heads'/name/'complete.json'; immutable(path, record)
            records.append(source.base.artifact(path))
        immutable(PUBLIC/'training_freeze.json', dict(registration_sha256=digest(PUBLIC/'registration.json'),
            CREATE_evidence=source.base.artifact(PUBLIC/'create_training_freeze.json'), groups=records,
            heads=216, updates=432000, held_outcomes_used=False, independent_roles_read=False))
        immutable(PUBLIC/'local_restore.json', dict(result_source='cached_verified', checkpoint_bytes=size,
            heads=216, archive_sha256=hashlib.sha256(r.stdout).hexdigest(),
            restore_code_sha256=digest(Path(__file__)), held_outcomes_used=False,
            new_training=False, local_inference='not_run', cross_architecture_retraining=False))
        print(json.dumps(dict(heads=216, bytes=size, restored='cached_verified', new_training=False)))


if __name__ == '__main__':
    main()
