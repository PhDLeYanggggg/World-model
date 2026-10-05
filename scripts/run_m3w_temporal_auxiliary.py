"""Registered matched auxiliary cost-head fitting; no inference-time future inputs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_temporal_target_audit as previous
from src.world_model import m3w_temporal_auxiliary as api

NAME = 'european_temporal_auxiliary_v1'
PUBLIC, PRIVATE = previous.PUBLIC.parent/NAME, previous.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
parent, sha, once = previous.parent, previous.sha, previous.once


def registration():
    cfg = json.loads(CONFIG.read_text())
    assert previous.registration() == json.loads(previous.REGISTRATION.read_text())
    assert sha(previous.PUBLIC/'verification.json') == cfg['parent_verification_sha256']
    receipt = json.loads((previous.PUBLIC/'verification.json').read_text())
    for key in ('summary', 'complete'):
        assert sha(previous.PUBLIC/(key+'.json')) == receipt[key+'_sha256']
    assert json.loads((previous.PUBLIC/'summary.json').read_text())['advance_to_auxiliary_design']
    paths = parent.forest.closure(ROOT, ['scripts.run_m3w_temporal_auxiliary'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_temporal_auxiliary.py']
    return cfg, dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in paths},
        parent_verification_sha256=cfg['parent_verification_sha256'],
        new_training_authorized=True, completed_training=False, independent_roles_read=False)


def beat(**kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    (PRIVATE/'heartbeat.json').write_text(json.dumps(row)+'\n')
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def storage(cfg):
    size = sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) if PRIVATE.exists() else 0
    if size > cfg['checkpoint_cap_bytes']:
        raise OSError('Registered checkpoint cap exceeded; preserve existing work')
    return api.storage_status(ROOT, reserve=cfg['disk_reserve_bytes'],
        remaining=cfg['checkpoint_cap_bytes']-size+cfg['atomic_checkpoint_headroom_bytes'])


def guard(cfg):
    result = storage(cfg)
    if not result['allowed']:
        raise OSError('No checkpoint-safe storage; preserve the last saved state: '+json.dumps(result))


def inputs(data, jobs, oid):
    docs = parent.parent.docs()
    for c in parent.parent.contexts(data, jobs, oid):
        for site in parent.inner.sources(c):
            group = c['name']+'_fit_'+site
            at, ids, x, env, y, _, upstream = parent.inner.training_arrays(c, data, site)
            tr, val, partition = parent.forest.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
            assert not set(data['recordings'][ids[tr]]) & set(data['recordings'][ids[val]])
            # The inherited reader reconstructs source labels; only TRAIN labels
            # and TRAIN-fitted preprocessing are passed to this optimizer.
            tid = ids[tr]
            series, _ = previous.api.targets(
                c['floor'][at][tr].astype(float)+data['origin'][tid, None],
                c['prediction'][at][tr].astype(float)+data['origin'][tid, None],
                data['target_eval'][tid], data['valid'][tid])
            pr = api.core.preprocess(x[tr], env[tr], y[tr], data['sites'][tid],
                data['recordings'][tid], data['frames'][tid], training_site=site)
            for seed in json.loads(CONFIG.read_text())['head_seeds']:
                source = docs[group, seed]
                assert sha(ROOT/source['checkpoint']['path']) == source['checkpoint']['sha256']
                state = joblib.load(ROOT/source['checkpoint']['path'])
                assert state['identity']['upstream'] == upstream and source['partition'] == partition
                api.core.exact(pr, state['preprocess'])
                identity = dict(group=group, source=site, seed=seed, upstream=upstream,
                    partition=partition, training_ids_sha256=api.forest.fingerprint(tid),
                    source_checkpoint=source['checkpoint'], registration_sha256=sha(PUBLIC/'registration.json'))
                yield identity, (x[tr], env[tr], y[tr], series, data['sites'][tid],
                                 data['recordings'][tid], data['frames'][tid], pr)


def inspect_first(identity, args, cfg):
    x, env, y, series, sites, rec, frames, pr = args
    known = np.isfinite(y).all(1)
    _, mask = previous.api.validate_series(series)
    assert np.array_equal(known, mask.any(1))
    mean = np.where(mask[..., None], series, 0).sum(1)/mask.sum(1).clip(1)[:, None]
    np.testing.assert_allclose(mean[known, 0], y[known, 1]-y[known, 0], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(mean[known, 1], y[known, 2], rtol=1e-7, atol=1e-7)
    groups, sg, _ = api.core.sampling.query_groups(sites, rec, frames, known)
    ix, seg, _ = api.core.sampling.draw_queries(groups, sg, cfg['head_training']['query_batch_size'],
                                               api.torch.Generator().manual_seed(identity['seed']+7919))
    z = api.torch.from_numpy(np.clip((x[ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
    e = api.torch.from_numpy((env[ix]/pr['scale']).astype(np.float32))
    target = api.torch.from_numpy((y[ix]/pr['scale']).astype(np.float32))
    measurements = {}
    for arm in api.ARMS:
        model = api.initialize(pr, cfg['head_training'], identity['seed'])
        p, a = model(z, e)
        loss = api.core.losses(p, target, api.torch.from_numpy(seg), cfg['head_training']['query_batch_size'], api.torch.from_numpy(pr['rms']))
        aux = api.temporal.auxiliary_loss(a, api.torch.from_numpy((api.auxiliary_targets(series[ix], arm)/pr['scale']).astype(np.float32)),
            api.torch.from_numpy(mask[ix]), api.torch.from_numpy(seg), cfg['head_training']['query_batch_size'])
        total = loss['total']+(0 if arm == 'none' else cfg['head_training']['auxiliary_weight'])*aux
        total.backward()
        assert all(param.grad is None or api.torch.isfinite(param.grad).all() for param in model.parameters())
        measurements[arm] = dict(initial_primary_loss=float(loss['total'].detach()),
            initial_auxiliary_loss=float(aux.detach()), finite_backward=True)
    return dict(identity=identity, rows=len(x), known_rows=int(known.sum()), sampled_rows=len(ix),
        arms=measurements, optimizer_updates=0, checkpoint_written=False, validation_labels_scored=False,
        engineering_only=True, result_source='fresh_run_real_TRAIN_forward_backward_not_model_training')


def train(cfg, iterator, *, pilot, resume):
    started = time.monotonic(); refs = []; fits = 0
    for identity, args in iterator:
        states = []
        for arm in api.ARMS:
            guard(cfg)
            key = identity['group']+'_head'+str(identity['seed'])+'_'+arm
            home = PRIVATE/'heads'/key; path = home/'checkpoint.pt.gz'; docpath = PUBLIC/'fits'/(key+'.json')
            def heartbeat(**kw):
                if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
                    raise TimeoutError('Registered 12-hour cap; resume last atomic checkpoint')
                beat(group=key, **kw)
            kwargs = dict(arm=arm, settings=cfg['head_training'], identity=identity, seed=identity['seed'],
                path=path, heartbeat=heartbeat, checkpoint_guard=lambda:guard(cfg))
            if pilot and arm == 'temporal' and not path.exists():
                api.fit(*args, **kwargs, stop_at=cfg['pilot_updates']//2)
            state = api.fit(*args, **kwargs, resume=resume or (pilot and arm == 'temporal'),
                stop_at=cfg['pilot_updates'] if pilot else None)
            states.append(state)
            if not pilot:
                doc = dict(identity=identity, arm=arm, step=state['step'], trace=state['trace'],
                    checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                    input_hashes=state['input_hashes'], draw_hash=state['draw_hash'],
                    source='fresh_run_neural_cost_head_training', validation_labels_scored=False,
                    new_forecaster_training=False, deployment_changed=False)
                once(docpath, doc); refs.append(dict(path=str(docpath.relative_to(ROOT)), sha256=sha(docpath)))
            fits += 1
        api.assert_matched(states)
        if pilot:
            replay_path = PRIVATE/'pilot_replay'/'checkpoint.pt.gz'
            replay = api.fit(*args, arm='temporal', settings=cfg['head_training'], identity=identity,
                seed=identity['seed'], path=replay_path, heartbeat=lambda **kw:beat(group='pilot_replay', **kw),
                checkpoint_guard=lambda:guard(cfg), resume=resume, stop_at=cfg['pilot_updates'])
            for k in replay:
                if k != 'seconds': api.core.exact(replay[k], states[-1][k])
            elapsed = time.monotonic()-started
            # Conservative estimate includes this pilot's load/replay overhead for every group.
            estimate = elapsed*cfg['source_heads']*cfg['head_training']['steps']/cfg['pilot_updates']
            once(PUBLIC/'pilot.json', dict(seconds=elapsed, fitted_arms=list(api.ARMS),
                optimizer_updates=cfg['pilot_updates'], exact_interrupted_resume=True, matched_draws=True,
                estimated_full_seconds=estimate, estimate_not_measurement=True,
                local_time_feasible=estimate<cfg['hard_runtime_limit_seconds'],
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                validation_labels_scored=False, real_TRAIN_data=True))
            return
    assert fits == cfg['neural_fits']
    once(PUBLIC/'training_freeze.json', dict(fits=refs, neural_fits=fits, seconds=time.monotonic()-started,
        validation_labels_scored=False, evaluation_status='not_run_separate_readout_required',
        new_forecasters=False, independent_roles_read=False, deployment_changed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['register', 'preflight', 'inspect', 'pilot', 'train'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg, reg = registration()
    if args.phase == 'register':
        once(PUBLIC/'registration.json', reg); print('Registered matched temporal auxiliary training'); return
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    parent.base.inter.committed(PUBLIC/'registration.json')
    if args.phase == 'preflight':
        result = dict(storage=storage(cfg), native_architecture=platform.machine(), cpu_threads=cfg['cpu_threads'],
            num_workers=cfg['num_workers'], scientific_training_run=False, remote_state='not_observed_by_this_command')
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        once(PUBLIC/('preflight_'+stamp+'.json'), result)
        print(json.dumps(result, indent=2)); return
    PRIVATE.mkdir(parents=True, exist_ok=True)
    api.torch.set_num_threads(cfg['cpu_threads']); api.torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase in ('pilot', 'train'): guard(cfg)
        if args.phase == 'pilot':
            assert not (PUBLIC/'pilot.json').exists(), 'Pilot already complete; reuse its verified receipt'
        if args.phase == 'train':
            assert not (PUBLIC/'training_freeze.json').exists(), 'Already frozen; do not retrain'
            pilot = json.loads((PUBLIC/'pilot.json').read_text())
            assert pilot['exact_interrupted_resume'] and pilot['local_time_feasible']
            assert pilot['peak_RSS_bytes'] < 40*2**30
        beat(state='loading_verified_source_assets', phase=args.phase)
        _, _, data, jobs, oid, _, _, _ = parent.inner.old.load()
        iterator = inputs(data, jobs, oid)
        if args.phase == 'inspect':
            identity, values = next(iterator)
            once(PUBLIC/'input_check.json', inspect_first(identity, values, cfg))
            beat(state='inspection_complete_no_optimizer_updates')
        else:
            train(cfg, iterator, pilot=args.phase == 'pilot', resume=args.resume)
            beat(state='training_phase_complete', phase=args.phase)


if __name__ == '__main__':
    main()
