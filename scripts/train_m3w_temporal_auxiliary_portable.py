"""Run the frozen temporal auxiliary optimizer on immutable TRAIN-only packets."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal

if __name__ == '__main__' and not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Allocated CREATE compute node required before numerical imports')

import numpy as np
from scripts import run_m3w_temporal_auxiliary as run

ARRAYS = ('x', 'envelope', 'target', 'series', 'sites', 'recordings', 'frames')


def packet(args):
    arrays = dict(zip(ARRAYS, args[:-1], strict=True))
    pr = args[-1]
    scalars = {k: v for k, v in pr.items() if not isinstance(v, np.ndarray)}
    arrays.update({'pr_'+k: v for k, v in pr.items() if isinstance(v, np.ndarray)})
    arrays['preprocess_json'] = np.array(json.dumps(scalars))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def unpack(content):
    with np.load(io.BytesIO(content), allow_pickle=False) as z:
        pr = json.loads(str(z['preprocess_json']))
        pr.update({k[3:]: z[k] for k in z.files if k.startswith('pr_')})
        return tuple(z[k] for k in ARRAYS)+(pr,)


def quota_storage(root, private, cfg, *, quota_free=None):
    if quota_free is None:
        home = Path('/users/k24101830')
        quota_free = int(os.getxattr(home, 'ceph.quota.max_bytes'))-int(os.getxattr(home, 'ceph.dir.rbytes'))
    size = sum(p.stat().st_size for p in private.rglob('*.pt.gz'))
    if size > cfg['checkpoint_cap_bytes']:
        raise OSError('Registered checkpoint cap exceeded')
    return run.api.storage_status(root, reserve=cfg['disk_reserve_bytes'], free=quota_free,
        remaining=cfg['checkpoint_cap_bytes']-size+cfg['atomic_checkpoint_headroom_bytes'])


def verified_iterator(root, manifest):
    for ref in manifest['heads']:
        path = root/'inputs'/(ref['group']+'.npz')
        content = path.read_bytes()
        if len(content) != ref['bytes'] or hashlib.sha256(content).hexdigest() != ref['sha256']:
            raise ValueError('TRAIN packet checksum mismatch')
        args = unpack(content)
        if ref['identity']['group'] != ref['group']:
            raise ValueError('Packet identity mismatch')
        yield ref['identity'], args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['pilot', 'train'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(); root = a.root.resolve()
    assert root.parent == Path('/users/k24101830/m3w') and root.name == run.NAME
    owner = json.loads((root/'.owner.json').read_text())
    assert owner['experiment'] == run.NAME
    cfg = json.loads((root/'config.json').read_text())
    manifest = json.loads((root/(a.phase+'_input_manifest.json')).read_text())
    assert manifest['input_role'] == 'source_TRAIN_only'
    assert not manifest['validation_rows_transferred'] and not manifest['independent_roles_read']
    assert manifest['registration_sha256'] == owner['registration_sha256']
    assert run.sha(root/'config.json') == manifest['config_sha256']
    for rel, digest in manifest['code_bindings'].items():
        assert run.sha(root/'code'/rel) == digest, 'Frozen execution code changed'
    assert len(manifest['heads']) == (1 if a.phase == 'pilot' else cfg['source_heads'])
    run.ROOT = root
    run.PRIVATE = root/'data/stage_cvpr2027_experiments'/run.NAME
    run.PUBLIC = root/'outputs/publication_readiness_2026_09'/run.NAME
    run.PRIVATE.mkdir(parents=True, exist_ok=True)
    run.PUBLIC.mkdir(parents=True, exist_ok=True)
    # Only the storage provider changes: personal Ceph quota, same 10 GiB reserve.
    run.storage = lambda config: quota_storage(root, run.PRIVATE, config)
    run.api.torch.set_num_threads(cfg['cpu_threads'])
    run.api.torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run.guard(cfg)
        if a.phase == 'train':
            pilot = json.loads((run.PUBLIC/'pilot.json').read_text())
            assert pilot['exact_interrupted_resume'] and pilot['matched_draws']
            assert pilot['local_time_feasible'] and pilot['peak_RSS_bytes'] < 14*2**30
            assert not (run.PUBLIC/'training_freeze.json').exists(), 'Do not refit a completed run'
        else:
            assert not (run.PUBLIC/'pilot.json').exists(), 'Reuse the completed pilot'
        run.beat(state='portable_phase_started', phase=a.phase, job_id=os.environ.get('SLURM_JOB_ID'),
                 optimizer_unchanged=True, validation_labels_scored=False)
        run.train(cfg, verified_iterator(root, manifest), pilot=a.phase == 'pilot', resume=a.resume)
        run.beat(state='portable_phase_complete', phase=a.phase)


if __name__ == '__main__':
    main()
