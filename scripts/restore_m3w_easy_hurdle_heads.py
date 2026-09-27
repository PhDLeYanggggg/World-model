"""Restore only own frozen CREATE checkpoints for unchanged local causal inference."""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from scripts import run_m3w_easy_hurdle as run
from scripts.prepare_m3w_create_runtime import HANDOFF
from scripts.verify_m3w_easy_hurdle_portable_training import check_state


def install_bundle(content, refs, destination, reserve):
    expected={r['path']:r['sha256'] for r in refs}
    if len(expected)!=len(refs):raise ValueError('Duplicate checkpoint references')
    with tarfile.open(fileobj=io.BytesIO(content),mode='r:') as tar:
        members=tar.getmembers()
        if len(members)!=len(expected) or {m.name for m in members}!=set(expected):
            raise ValueError('Only the exact frozen checkpoint set is accepted')
        values=[]
        for member in members:
            p=PurePosixPath(member.name)
            if (not member.isfile() or p.is_absolute() or '..' in p.parts or len(p.parts)!=4
                    or p.parts[0]!='heads' or p.parts[2] not in run.api.ARMS or p.parts[3]!='checkpoint.pt.gz'):
                raise ValueError('Unsafe or unrelated archive member')
            data=tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest()!=expected[member.name]:raise ValueError('Checkpoint checksum mismatch')
            values.append((destination/member.name,data))
        for path,data in values:
            if path.exists():
                if path.read_bytes()!=data:raise ValueError('Existing local checkpoint differs; do not overwrite')
                continue
            reserve(len(data));path.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.restore-',delete=False) as f:
                temp=Path(f.name)
                try:
                    f.write(data);f.flush();os.fsync(f.fileno())
                except BaseException:
                    temp.unlink(missing_ok=True);raise
            os.replace(temp,path)
    return sum(len(data) for _,data in values)


def main():
    run.base.torch.set_num_threads(4);run.base.torch.set_num_interop_threads(1)
    cfg,ident=run.identity()
    frozen_path=run.PUBLIC/'create_training_freeze.json';audit_path=run.PUBLIC/'create_training_verification.json'
    for path in (frozen_path,audit_path):run.base.inter.committed(path)
    frozen=json.loads(frozen_path.read_text());audit=json.loads(audit_path.read_text());trained=frozen['receipt']
    assert frozen['registration']==run.base.artifact(run.PUBLIC/'registration.json')==audit['registration']
    assert trained['groups']==108 and len(trained['artifacts'])==216
    assert audit['receipt']['training_receipt_sha256']==frozen['receipt_sha256']
    assert audit['receipt']['first_group_full_training_replay_exact_except_elapsed']
    manifest=json.loads((run.PRIVATE/'remote_input_manifest.json').read_text())
    assert manifest['registration']==ident
    assert trained['manifest_sha256']==audit['receipt']['manifest_sha256']
    free=shutil.disk_usage(run.PRIVATE).free;projection=frozen['checkpoint_bytes']+600_000_000
    if free-projection<cfg['disk_reserve_bytes']:
        raise OSError('Checkpoint plus conservative action/report projection crosses unchanged10GiB reserve')
    ssh=json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
    code=r'''
import hashlib,io,json,pathlib,sys,tarfile
p=json.loads(sys.stdin.read());home=pathlib.Path(p['home'])
assert json.loads((home/'.owner.json').read_text())['experiment']=='european_easy_hurdle_v1'
for name,sha in [('training_complete.json',p['trained_sha256']),('training_audit.json',p['audit_sha256'])]:
    assert hashlib.sha256((home/name).read_bytes()).hexdigest()==sha
r=json.loads((home/'training_complete.json').read_text());assert r['complete'] and r['groups']==108
assert r['artifacts']==p['refs']
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
    for ref in r['artifacts']:
        rel=pathlib.PurePosixPath(ref['path'])
        assert rel==pathlib.PurePosixPath('heads')/ref['group']/ref['arm']/'checkpoint.pt.gz'
        assert not rel.is_absolute() and '..' not in rel.parts
        data=(home/rel).read_bytes();assert hashlib.sha256(data).hexdigest()==ref['sha256']
        entry=tarfile.TarInfo(str(rel));entry.size=len(data);entry.mode=0o600
        tar.addfile(entry,io.BytesIO(data))
'''
    with (run.PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        payload=dict(home=manifest['remote_path'],trained_sha256=frozen['receipt_sha256'],
                     audit_sha256=audit['receipt_sha256'],refs=trained['artifacts'])
        r=subprocess.run(ssh+[shlex.join(['/usr/bin/python3','-c',code])],input=json.dumps(payload).encode(),capture_output=True,timeout=180)
        if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1500:])
        def reserve(needed):
            if shutil.disk_usage(run.PRIVATE).free-needed<cfg['disk_reserve_bytes']:
                raise OSError('Preserve10GiB; keep completed files for verified resume')
        size=install_bundle(r.stdout,trained['artifacts'],run.PRIVATE,reserve)
        assert size==frozen['checkpoint_bytes']
        parent_fits={Path(ref['path']).parent.name:ref for ref in json.loads((run.parent.PUBLIC/'training_freeze.json').read_text())['groups']}
        refs=[];first=None
        for group in manifest['groups']:
            name=group['group'];home=run.PRIVATE/'heads'/name;states={}
            for arm in run.api.ARMS:
                state=run.api.head.read_checkpoint(home/arm/'checkpoint.pt.gz')
                identity=state['identity'];assert identity['registration']==frozen['registration']
                assert identity['parent_fit']==parent_fits[name]
                run.checked(identity['parent_fit'])
                check_state(state,identity,cfg,arm);states[arm]=state
            run.api.assert_matched(states['marginal'],states['supervised'])
            assert states['marginal']['identity']==states['supervised']['identity']
            identity=states['marginal']['identity'];roles=identity['roles']
            run.base.floor_api.api.assert_roles(roles['producer_sites'],roles['controller_sites'],roles['training_sites'],roles['held_sites'])
            record=dict(identity=identity,arms=list(run.api.ARMS),matched_initialization_and_queries=True,
                held_outcomes_used=False,artifacts={arm:run.base.artifact(home/arm/'checkpoint.pt.gz') for arm in run.api.ARMS})
            run.base.immutable_json(home/'complete.json',record);refs.append(run.base.artifact(home/'complete.json'))
            if first is None:first=(name,identity)
        run.base.immutable_json(run.PUBLIC/'training_freeze.json',dict(identity=ident,groups=refs,heads=216,
            updates=432000,held_outcomes_used=False,independent_roles_read=False))
        run.base.immutable_json(run.PUBLIC/'fit_replay.json',dict(group=first[0],heads=2,updates=4000,
            exact_except_elapsed=True,input_identity=first[1],execution_location='CREATE_same_runtime',
            source='cached_verified_CREATE_full_training_replay',evidence=run.base.artifact(audit_path),
            cross_architecture_training_replay=False))
        run.base.immutable_json(run.PUBLIC/'local_restore.json',dict(result_source='cached_verified_checkpoint_restore',
            checkpoint_bytes=size,heads=216,free_before_bytes=free,free_after_bytes=shutil.disk_usage(run.PRIVATE).free,
            projection_bytes=projection,disk_reserve_bytes=cfg['disk_reserve_bytes'],remote_training=run.base.artifact(frozen_path),
            remote_verification=run.base.artifact(audit_path),archive_sha256=hashlib.sha256(r.stdout).hexdigest(),
            held_outcomes_used=False,new_training=False,local_inference='not_run',
            adapter=run.base.artifact(Path(__file__).resolve())))
        run.beat('frozen_heads_restored',heads=216,bytes=size)


if __name__=='__main__':main()
