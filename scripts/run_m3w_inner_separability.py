"""Registered single-locality fit and locality-held development learning control."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import diagnose_m3w_fitting_switches as parent
from src.world_model import m3w_inner_separability as api
from src.world_model.m3w_easy_component_diagnostic import weights

old, base = parent.old, parent.base
NAME = 'european_inner_separability_v1'
PUBLIC = parent.PUBLIC.parent/NAME; PRIVATE = parent.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
digest, immutable = parent.digest, parent.immutable


def registration():
    cfg = json.loads(CONFIG.read_text()); sp = parent.PUBLIC/'verification.json'
    assert digest(sp) == cfg['parent_seal_sha256']; seal = json.loads(sp.read_text())
    for p, h in seal['source_bindings'].items(): assert digest(ROOT/p) == h, p
    for p, h in seal['artifacts'].items(): assert digest(parent.PUBLIC/p) == h, p
    assert cfg['arms'] == list(api.ARMS) and cfg['heads'] == 144 and cfg['risk_budget'] == .02
    assert not any(cfg[k] for k in ('outer_held_outcomes_used', 'independent_roles_read', 'threshold_search',
        'new_forecasters', 'deployment_changed', 'formal_primary_replaced', 'stage5c_executed', 'smc_enabled'))
    paths = parent.source.closure(ROOT, ['scripts.run_m3w_inner_separability'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_inner_separability.py']
    return cfg, dict(parent_seal=base.artifact(sp), bindings={str(p.relative_to(ROOT)): digest(p) for p in paths},
        unique_fits=72, heads=144, directional_views=216, parameter_updates=288000,
        primary='nonlinear_minus_affine_signed_score_MSE', independent_roles_read=False)


def beat(state, **kw):
    row = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    base.inter.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def guard(extra=32*2**20):
    if shutil.disk_usage(PRIVATE).free < 10*2**30+extra: raise OSError('Keep10GiB and completed work')


def checked(ref):
    assert base.artifact(ROOT/ref['path']) == ref
    return json.loads((ROOT/ref['path']).read_text())


def sources(c):
    sites = sorted({s for fit, _ in c['pairs'] for s in fit})
    assert len(sites) == 4 and not set(sites) & (set(c['producer_sites']) | set(c['controller_sites']))
    for fit, held in c['pairs']: base.floor_api.api.assert_roles(c['producer_sites'], c['controller_sites'], fit, held)
    return sites


def training_arrays(c, data, site):
    assert site in sources(c)
    at = np.flatnonzero(data['sites'][c['ids']] == site); ids = c['ids'][at]
    cv, _, (floor, _), (neural, _) = base.floor_api.costs(c, data, at)
    y = api.targets(cv, floor, neural, c['job']['design']['easy_cut'])
    x, env = c['x'][at], c['env'][at]
    assert x.shape[1] == 380
    pr = api.preprocess(x, env, y, data['sites'][ids], data['recordings'][ids], data['frames'][ids], training_site=site)
    ident = dict(registration_sha256=digest(PUBLIC/'registration.json'), context=c['name'], train_site=site,
        producer_sites=c['producer_sites'], controller_sites=c['controller_sites'],
        upstream_context=c['job']['old_identity'], seed=c['job']['old_identity']['seed'],
        fitting_ids_hash=base.inter.array_hash(ids), input_hashes={k: base.inter.array_hash(v)
            for k, v in dict(x=x, envelope=env, targets=y, frames=data['frames'][ids]).items()},
        other_fitting_source_used=False, outer_held_outcomes_used=False)
    return at, ids, x, env, y, pr, ident


def train(cfg, data, jobs, oid, *, pilot=False, replay=False, resume=False):
    receipt = PUBLIC/('fit_replay.json' if replay else 'training_freeze.json')
    if receipt.exists(): raise ValueError('Completed receipt exists')
    if not pilot and not replay:
        p = json.loads((PUBLIC/'pilot.json').read_text()); assert p['local_feasible'] and p['storage_sufficient']
    start = time.monotonic(); records = []; causal = {k: data[k] for k in old.parent.CAUSAL_KEYS}
    for c in base.floor_api.contexts(causal, jobs, oid):
        for site in sources(c):
            guard(); name = c['name']+'_fit_'+site; home = PRIVATE/'heads'/name; path = home/'complete.json'
            if path.exists() and resume and not replay:
                doc = json.loads(path.read_text()); assert doc['identity']['registration_sha256'] == digest(PUBLIC/'registration.json')
                for r in doc['artifacts'].values(): assert base.artifact(ROOT/r['path']) == r
            else:
                if path.exists() and not replay: raise ValueError('Existing fit requires resume')
                at, ids, x, env, y, pr, identity = training_arrays(c, data, site); states = {}; began = time.monotonic()
                for arm in api.ARMS:
                    cp = (PRIVATE/'fit_replay'/name if replay else home)/arm/'checkpoint.pt.gz'
                    states[arm] = api.fit(x, env, y, data['sites'][ids], data['recordings'][ids], data['frames'][ids], pr,
                        arm=arm, settings=cfg['head_training'], identity=identity, seed=identity['seed'], path=cp,
                        heartbeat=lambda **kw: beat(group=name, arm=arm, **kw), resume=resume and not replay,
                        stop_at=cfg['pilot_updates'] if pilot else None)
                    if replay:
                        original = api.read_checkpoint(home/arm/'checkpoint.pt.gz')
                        for k in states[arm]:
                            if k != 'seconds': api.exact(original[k], states[arm][k])
                api.assert_matched(states['affine'], states['nonlinear'])
                initial = [api.predict(s, x[:128], env[:128], initial=True)[0] for s in states.values()]
                np.testing.assert_array_equal(*initial)
                if pilot:
                    bytes_ = sum((home/a/'checkpoint.pt.gz').stat().st_size for a in api.ARMS)
                    projected = bytes_*72*3+128*2**20; elapsed = time.monotonic()-began
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    if platform.system() != 'Darwin': rss *= 1024
                    immutable(PUBLIC/'pilot.json', dict(group=name, updates_per_head=100, paired_seconds=elapsed,
                        projected_full_seconds=elapsed*20*72, projection_not_measured_runtime=True,
                        checkpoint_bytes=bytes_, projected_bytes=projected, peak_RSS_bytes=rss,
                        local_feasible=elapsed*20*72 < 12*3600 and rss < 40*2**30,
                        storage_sufficient=shutil.disk_usage(PRIVATE).free-projected > 10*2**30,
                        machine=platform.machine(), torch=api.torch.__version__, threads=4, workers=0))
                    return
                if replay:
                    immutable(receipt, dict(group=name, heads=2, updates=4000, exact_except_elapsed=True,
                        all72_pairs_retrained=False, seconds=time.monotonic()-start)); return
                doc = dict(identity=identity, artifacts={a: base.artifact(home/a/'checkpoint.pt.gz') for a in api.ARMS},
                    matching_queries=True, equal_initial_predictions=True,
                    fitting_monitor={a: s['trace'][-1]['monitor'] for a, s in states.items()},
                    parameter_counts={a: sum(v.numel() for v in s['model'].values()) for a, s in states.items()})
                immutable(path, doc)
            records.append(base.artifact(path)); beat('fit_complete', fits=len(records), group=name)
    assert len(records) == 72
    immutable(receipt, dict(groups=records, heads=144, parameter_updates=288000, seconds=time.monotonic()-start,
        outer_held_outcomes_used=False, independent_roles_read=False))
    beat('training_complete', fits=72, seconds=time.monotonic()-start)


def view_predictions(c, causal, pair, train_site, fits):
    if set(causal) != set(old.parent.CAUSAL_KEYS): raise ValueError('Causal whitelist required')
    fitting, outer = c['pairs'][pair]; target_site = next(s for s in fitting if s != train_site)
    assert train_site in fitting and train_site != target_site and target_site not in outer
    ref = fits[c['name']+'_fit_'+train_site]; doc = checked(ref)
    assert doc['identity']['train_site'] == train_site
    pos = np.flatnonzero(causal['sites'][c['ids']] == target_site); ids = c['ids'][pos]
    values = {}; states = {}; support = None
    for arm, cp in doc['artifacts'].items():
        assert base.artifact(ROOT/cp['path']) == cp
        state = api.read_checkpoint(ROOT/cp['path']); states[arm] = state
        assert state['identity'] == doc['identity'] and state['step'] == 2000
        p, ok = api.predict(state, c['x'][pos], c['env'][pos]); values[arm] = p
        if support is not None: np.testing.assert_array_equal(support, ok)
        support = ok
    api.assert_matched(states['affine'], states['nonlinear'])
    values['intercept'] = api.predict(states['affine'], c['x'][pos], c['env'][pos], initial=True)[0]
    q = api.signed(values['intercept'])
    actions = api.decisions({k: values[k] for k in api.ARMS}, c['moving'][pos], support,
        causal['recordings'][ids], causal['frames'][ids], ids)
    actions['intercept'] = c['moving'][pos] & support & (q[:, 0] > 0) & (q[:, 1:] <= 0).all(1)
    arrays = dict(ids=ids, support=support, **actions)
    meta = dict(context=c['name'], pair=pair, train_site=train_site, eval_site=target_site,
        outer_held_sites=outer, fit_ref=ref, causal_only=True, outer_held_outcomes_used=False,
        score_hashes={a: base.inter.array_hash(p) for a, p in values.items()})
    return pos, values, arrays, meta, states['affine']['preprocess']


def decide(cfg, data, jobs, oid, *, replay=False, resume=False):
    base.inter.committed(PUBLIC/'training_freeze.json'); start = time.monotonic()
    fits = {Path(r['path']).parent.name: r for r in json.loads((PUBLIC/'training_freeze.json').read_text())['groups']}
    refs = []; causal = {k: data[k] for k in old.parent.CAUSAL_KEYS}
    for c in base.floor_api.contexts(causal, jobs, oid):
        for pair, (fitting, _) in enumerate(c['pairs']):
            for site in fitting:
                guard(); name = c['name']+f'_pair{pair}_from_'+site; path = PRIVATE/'decisions'/(name+'.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and resume and not replay:
                    doc = json.loads(path.read_text()); assert base.artifact(ROOT/doc['arrays']['path']) == doc['arrays']
                else:
                    if path.exists() and not replay: raise ValueError('Existing action needs resume')
                    _, p, arrays, meta, _ = view_predictions(c, causal, pair, site, fits)
                    parent.source.labels.write_or_compare(path.with_suffix('.npz'), arrays, replay=replay)
                    doc = dict(registration_sha256=digest(PUBLIC/'registration.json'), **meta,
                        arrays=base.artifact(path.with_suffix('.npz')))
                    immutable(path, doc)
                refs.append(base.artifact(path))
        beat('actions_replayed' if replay else 'actions_frozen', views=len(refs), context=c['name'])
    assert len(refs) == 216
    receipt = PUBLIC/('decision_replay.json' if replay else 'decision_freeze.json')
    immutable(receipt, dict(groups=refs, exact=replay, seconds=time.monotonic()-start,
        outer_held_outcomes_used=False, independent_roles_read=False))


def evaluate(cfg, data, jobs, oid, *, replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json'); start = time.monotonic()
    fits = {Path(r['path']).parent.name: r for r in json.loads((PUBLIC/'training_freeze.json').read_text())['groups']}
    rows = []; quality = []; causal = {k: data[k] for k in old.parent.CAUSAL_KEYS}
    for c in base.floor_api.contexts(causal, jobs, oid):
        for pair, (fitting, _) in enumerate(c['pairs']):
            for site in fitting:
                name = c['name']+f'_pair{pair}_from_'+site; path = PRIVATE/'decisions'/(name+'.json')
                doc = json.loads(path.read_text()); assert base.artifact(ROOT/doc['arrays']['path']) == doc['arrays']
                pos, p, arrays, meta, pr = view_predictions(c, causal, pair, site, fits)
                assert meta['score_hashes'] == doc['score_hashes']
                parent.source.labels.write_or_compare(path.with_suffix('.npz'), arrays, replay=True)
                ids = arrays['ids']; cv, cf, (floor, ff), (neural, nf) = base.floor_api.costs(c, data, pos)
                y = api.targets(cv, floor, neural, c['job']['design']['easy_cut']); known = np.isfinite(y).all(1)
                w, _ = weights(data['sites'][ids], data['recordings'][ids], data['frames'][ids], known)
                for arm in ('intercept', *api.ARMS):
                    d = (api.signed(p[arm][known])-api.signed(y[known]))/pr['scale']/pr['rms'][5:]
                    quality.append(dict(view=name, site=meta['eval_site'], train_site=site, arm=arm,
                        seed=c['job']['old_identity']['seed'], signed_MSE=(w[known, None]*d**2).sum(0).tolist(),
                        mean_signed_MSE=float((w[known, None]*d**2).sum(0).mean())))
                takes = {k: arrays[k] for k in ('intercept', *api.ARMS, *[a+'_matched' for a in api.ARMS])}
                takes.update(floor=np.zeros(len(ids), bool), neural_unprotected=np.ones(len(ids), bool))
                for policy, take in takes.items():
                    m = base.floor_api.metric(cv, floor, neural, cf, ff, nf, np.where(take, neural, floor),
                        np.where(take, nf, ff), take, data['valid'][ids], c['job']['design']['easy_cut'], c['job']['design']['hard_cut'])
                    easy = known & (cv > 0) & (cv <= c['job']['design']['easy_cut']); chosen = take & known
                    ref = float(floor[chosen & easy].sum()); harm = np.maximum(neural-floor, 0)
                    m['selected_easy_positive_harm_ratio'] = float(harm[chosen & easy].sum())/ref if ref > 0 else None
                    rows.append(dict(view=name, site=meta['eval_site'], train_site=site, policy=policy,
                        seed=c['job']['old_identity']['seed'], metric=m))
        beat('internal_readout', directional_views=len(quality)//3, context=c['name'])
    result = dict(rows=rows, quality=quality, primary='nonlinear_minus_affine_mean_signed_MSE',
        result_source='fresh_run_internal_locality_held_development', independent_confirmation=False,
        outer_held_outcomes_used=False, models_selected=False, deployment_changed=False)
    assert len(quality) == 648 and len(rows) == 1512
    immutable(PUBLIC/'readout.json', result)
    immutable(PUBLIC/('evaluation_replay.json' if replay else 'evaluation_runtime.json'),
        dict(exact=replay, views=216, seconds=time.monotonic()-start))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register','pilot','train','replay_fit','decide','replay_decide','evaluate','replay_evaluate'])
    p.add_argument('--resume', action='store_true'); a = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    cfg, reg = registration()
    if a.phase == 'register': immutable(PUBLIC/'registration.json', reg); print(json.dumps(dict(registered=True))); return
    assert reg == json.loads((PUBLIC/'registration.json').read_text()); base.inter.committed(PUBLIC/'registration.json')
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guard(); beat('loading_registered_inputs', phase=a.phase)
        _, _, data, jobs, oid, _, _, _ = old.load()
        if a.phase in ('pilot','train','replay_fit'):
            train(cfg,data,jobs,oid,pilot=a.phase=='pilot',replay=a.phase=='replay_fit',resume=a.resume)
        elif a.phase in ('decide','replay_decide'):
            decide(cfg,data,jobs,oid,replay=a.phase=='replay_decide',resume=a.resume)
        else: evaluate(cfg,data,jobs,oid,replay=a.phase=='replay_evaluate')


if __name__ == '__main__': main()
