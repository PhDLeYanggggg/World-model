"""Replay frozen heads on their original TRAIN rows, without optimizer updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = 'european_temporal_auxiliary_v1'
REL = Path('outputs/publication_readiness_2026_09')/NAME
PRIVATE = Path('data/stage_cvpr2027_experiments')/NAME
DIAG = 'false_safe_train_diagnostic_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Preserve conflicting diagnostic receipt')
    else:
        with path.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def verify_movement():
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise RuntimeError('Native arm64 required before numerical imports')
    import numpy as np
    from scripts import run_m3w_temporal_auxiliary as run
    from src.world_model.m3w_false_safe_diagnostic import moving_from_features
    run.api.torch.set_num_threads(4); run.api.torch.set_num_interop_threads(1)
    manifest = json.loads((ROOT/PRIVATE/'create_train_input_manifest.json').read_text())
    groups = {r['group']: r for r in manifest['heads']}
    docs = {json.loads((ROOT/ref['path']).read_text())['identity']['group']:
            json.loads((ROOT/ref['path']).read_text())
            for ref in json.loads((ROOT/REL/'training_freeze.json').read_text())['fits']}
    _, _, data, jobs, oid, _, _, _ = run.parent.inner.old.load()
    checks = []
    for c in run.parent.parent.contexts(data, jobs, oid):
        for site in run.parent.inner.sources(c):
            group = c['name']+'_fit_'+site
            at, ids, x, env, y, _, upstream = run.parent.inner.training_arrays(c, data, site)
            tr, val, partition = run.parent.forest.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
            assert partition == groups[group]['identity']['partition']
            assert not set(data['recordings'][ids[tr]]) & set(data['recordings'][ids[val]])
            expected = c['moving'][at][tr]
            actual = moving_from_features(x[tr])
            np.testing.assert_array_equal(expected, actual)
            h = run.api.forest.fingerprint(x[tr])
            assert h == docs[group]['input_hashes']['x']
            assert run.api.forest.fingerprint(ids[tr]) == groups[group]['identity']['training_ids_sha256']
            checks.append(dict(group=group, train_rows=len(actual), moving=int(actual.sum()),
                mismatches=0, x_sha256=h, moving_sha256=run.api.forest.fingerprint(actual),
                packet_sha256=groups[group]['sha256']))
            print(json.dumps(checks[-1]), flush=True)
    assert len(checks) == 24 and {r['group'] for r in checks} == set(groups)
    once(ROOT/REL/DIAG/'movement_verification.json', dict(result_source='fresh_run_TRAIN_causal_mask_verification',
        groups=checks, training_freeze_sha256=sha(ROOT/REL/'training_freeze.json'),
        manifest_sha256=sha(ROOT/PRIVATE/'create_train_input_manifest.json'),
        new_validation_predictions=False, source_loader_reads_existing_exposed_arrays=True,
        independent_roles_read=False, used_future_to_derive_moving=False))


def run_remote(root):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated compute required before numerical imports')
    from scripts.train_m3w_temporal_shards import verify
    root, cfg, manifest, _, _ = verify(root)
    home = root/REL/DIAG
    reg = json.loads((root/(DIAG+'_registration.json')).read_text())
    for rel, h in reg['bindings'].items():
        assert sha(root/'code'/rel) == h
    assert sha(root/REL/'training_freeze.json') == reg['training_freeze_sha256']
    assert sha(root/'train_input_manifest.json') == reg['manifest_sha256']
    from scripts import train_m3w_temporal_auxiliary_portable as port
    from src.world_model import m3w_false_safe_diagnostic as api
    import numpy as np
    port.run.api.torch.set_num_threads(4); port.run.api.torch.set_num_interop_threads(1)
    movements = {r['group']: r for r in reg['movement']['groups']}
    fits = {}
    for ref in json.loads((root/REL/'training_freeze.json').read_text())['fits']:
        assert sha(root/ref['path']) == ref['sha256']
        doc = json.loads((root/ref['path']).read_text())
        key = doc['identity']['group'], doc['identity']['seed'], doc['arm']
        assert key not in fits and doc['step'] == 2000
        fits[key] = doc
    assert len(fits) == 216
    refs = []; previous_group = None; started = time.monotonic()
    for item in manifest['heads']:
        group, identity = item['group'], item['identity']
        if previous_group != group:
            packet = root/'inputs'/(group+'.npz')
            assert sha(packet) == item['sha256'] and packet.stat().st_size == item['bytes']
            x, env, y, series, sites, rec, frames, pr = port.unpack(packet.read_bytes())
            moving = api.moving_from_features(x)
            fingerprint = port.run.api.forest.fingerprint
            assert fingerprint(moving) == movements[group]['moving_sha256']
            assert item['sha256'] == movements[group]['packet_sha256']
            hashes = {k: fingerprint(np.asarray(a)) for k, a in dict(x=x, envelope=env,
                primary=y, temporal=series, sites=sites, recordings=rec, frames=frames).items()}
            previous_group = group
        states = []
        for arm in port.run.api.ARMS:
            doc = fits[group, identity['seed'], arm]; cp = root/doc['checkpoint']['path']
            assert cp.resolve().is_relative_to(root/PRIVATE/'heads')
            assert sha(cp) == doc['checkpoint']['sha256'] and cp.stat().st_size == doc['checkpoint']['bytes']
            state = port.run.api.core.read_checkpoint(cp)
            for k, v in dict(identity=identity, arm=arm, input_hashes=hashes, step=2000,
                             preprocess=pr, settings=cfg['head_training'], seed=identity['seed']).items():
                port.run.api.core.exact(state[k], v)
            states.append(state)
        port.run.api.assert_matched(states)
        for state in states:
            arm = state['arm']; doc = fits[group, identity['seed'], arm]
            key = group+'_head'+str(identity['seed'])+'_'+arm
            pred, _, support = port.run.api.predict(state, x, env)
            repeated, _, repeated_support = port.run.api.predict(state, x, env)
            np.testing.assert_array_equal(pred, repeated); np.testing.assert_array_equal(support, repeated_support)
            result = api.diagnose(pred, y, moving, support, rec, frames, pr['scale'])
            value = dict(identity=identity, arm=arm, checkpoint_sha256=doc['checkpoint']['sha256'],
                prediction_sha256=fingerprint(pred), source_packet_sha256=item['sha256'], result=result,
                result_source='fresh_run_frozen_model_TRAIN_resubstitution', optimizer_updates=0,
                validation_scored=False, independent_roles_read=False, thresholds_changed=False)
            path = home/'heads'/(key+'.json'); once(path, value)
            refs.append(dict(path=str(path.relative_to(root)), sha256=sha(path)))
            beat = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), job_id=os.environ['SLURM_JOB_ID'],
                state='TRAIN_diagnostic', heads=len(refs), total=216, key=key, elapsed=time.monotonic()-started)
            home.mkdir(parents=True, exist_ok=True)
            (home/'heartbeat.json').write_text(json.dumps(beat)+'\n')
            print(json.dumps(beat), flush=True)
    assert len(refs) == 216
    once(home/'complete.json', dict(result_source='fresh_run_TRAIN_only_diagnostic', heads=refs,
        unique_packets=24, total_heads=216, job_id=os.environ['SLURM_JOB_ID'],
        registration_sha256=sha(root/(DIAG+'_registration.json')),
        elapsed_seconds=time.monotonic()-started, optimizer_updates=0,
        validation_scored=False, independent_roles_read=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['movement', 'run']); p.add_argument('--root', type=Path)
    a = p.parse_args()
    if a.mode == 'movement':
        verify_movement()
    else:
        if a.root is None: p.error('--root required')
        run_remote(a.root)


if __name__ == '__main__':
    main()
