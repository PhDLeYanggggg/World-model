"""Matched causal inputs, different policy references; no readout selection."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_fixed_producer_roles as prior
from src.world_model.m3w_incumbent_relative import features, targets, choices, replay
import numpy as np
import torch

base = prior.base
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_incumbent_relative_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_incumbent_relative_v1'
CONFIG = 'configs/m3w_european_incumbent_relative_v1.json'
FILES = (CONFIG, 'src/world_model/m3w_incumbent_relative.py',
    'scripts/run_m3w_european_incumbent_relative.py', 'tests/test_m3w_incumbent_relative.py',
    'outputs/publication_readiness_2026_09/european_incumbent_relative_v1/registration.md')
artifact, digest, immutable_json, array_hash = prior.artifact, prior.digest, prior.immutable_json, prior.array_hash


def beat(state, **kw):
    r = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    base.cross.json_write(PRIVATE/'heartbeat.json', r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r), flush=True)


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    assert cfg['arms'] == ['floor_reference', 'incumbent_reference']
    assert cfg['policies'] == ['floor_reference', 'incumbent_reference', 'add_only', 'remove_only',
        'ridge_incumbent', 'old_stop', 'previous_matched', 'raw_neural']
    assert (cfg['groups'], cfg['views'], cfg['new_neural_heads'], cfg['ridge_fits'], cfg['neural_updates']) == (36, 288, 144, 72, 288000)
    assert (cfg['risk_budget'], cfg['ridge_alpha'], cfg['bootstrap_resamples'], cfg['bootstrap_seed']) == (.02, .01, 3000, 39271)
    assert not any(cfg[k] for k in ('new_forecaster_training', 'threshold_refit', 'calibration_refit',
        'reserved_roles_opened', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    assert digest(prior.PUBLIC/'summary_metrics.json') == cfg['prior_summary_sha256']
    done = json.loads((prior.PUBLIC/'completion_checks.json').read_text()); assert done['all_passed']
    for f, sha in done['artifact_hashes'].items(): assert digest(prior.PUBLIC/f) == sha
    prior.ensure_frozen()
    pcfg, bcfg, ctx, bid, pid = prior.load()
    assert cfg['head_training'] == pcfg['head_training'] and cfg['seeds'] == pcfg['seeds'] and cfg['events'] == pcfg['events']
    identity = dict(bindings={f: digest(ROOT/f) for f in FILES}, parent=pid,
        previous_completion=artifact(prior.PUBLIC/'completion_checks.json'), rosters=pid['rosters'])
    immutable_json(PRIVATE/'identity.json', identity)
    return cfg, bcfg, ctx, bid, pid, identity


def groups(bcfg, ctx, bid, pid):
    yield from prior.groups(bcfg, ctx, bid, pid)


def incumbent(g, data, bid, ids):
    copy = dict(g, roles=dict(g['roles'], readout=ids))
    return prior.old_scores(copy, data, bid)[1]


def prepare():
    cfg, bcfg, ctx, bid, pid, identity = load(); data = ctx[2]; rows = []
    prior.checked_assets(pid)
    for g in groups(bcfg, ctx, bid, pid):
        r = dict(group=g['name'], roles={}, easy_cut=g['design']['easy_cut'], hard_cut=g['design']['hard_cut'])
        for role in ('controller', 'readout'):
            ids = g['roles'][role]; old = incumbent(g, data, bid, ids)
            x = features(g['x'][ids], old)
            r['roles'][role] = dict(rows=len(ids), ids_sha256=array_hash(ids), choice_sha256=array_hash(old), input_sha256=array_hash(x))
        rows.append(r); beat('assets_verified', group=g['name'])
    assert len(rows) == 36
    immutable_json(PRIVATE/'assets_complete.json', dict(identity=identity, groups=rows, all_passed=True,
        new_old_head_replays=144, original_choices_replayed=72, cached_neural_verification=artifact(prior.PRIVATE/'assets_complete.json')))


def checked_assets(identity):
    r = json.loads((PRIVATE/'assets_complete.json').read_text())
    assert r['identity'] == identity and r['all_passed'] and len(r['groups']) == 36
    assert artifact(ROOT/r['cached_neural_verification']['path']) == r['cached_neural_verification']


def fit_one(x, y, sites, cv, env, tx, tenv, ids, directory, hid, cfg, seed, task, resume, pilot):
    if (directory/'complete.json').exists(): return base.checked(directory, hid)
    if shutil.disk_usage(PRIVATE).free < 10*1024**3: raise OSError('Below10GiB; preserve checkpoints')
    pr = base.ordinary.preprocess(x, y, cv, sites, '__excluded_outer__')
    kw = dict(seed=seed, settings=cfg['head_training'], identity=hid, directory=directory, resume=resume,
        stop_at=100 if pilot else None, heartbeat=lambda **v: beat(head=directory.name, **v))
    if task == 'utility':
        model, fit = base.geometric.fit(x, y, sites, env, pr, task=task, **kw)
        pred = lambda xx, ee: base.geometric.predict(model, xx, ee, pr); kind = 'bounded_utility'
    else:
        scale = prior.fixed_rank_scale(y, sites, pr, seed=seed, batch_size=256, batches=40)
        model, fit = base.ranked.fit(x, y, sites, env, pr, rank_weight=1., epsilon=1e-6, fixed_denominator=scale, **kw)
        pred = lambda xx, ee: base.hurdle.predict(model, xx, ee, pr); kind = 'ranked_risk'
    if pilot: return dict(identity=hid, fit=fit, checkpoint=artifact(directory/'checkpoint.pt'))
    scores = pred(tx, tenv); path = directory/'scores.npz'; tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, ids=ids, scores=scores); os.replace(tmp, path)
    r = dict(identity=hid, fit=fit, kind=kind, replay_exact=True,
        artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'), scores=artifact(path)))
    fresh, _ = prior.previous.restored(r); n = min(len(ids), cfg['replay_rows'])
    np.testing.assert_array_equal(fresh(tx[:n], tenv[:n]), scores[:n])
    assert fit['unknown_rows_sampled'] == 0
    r['prefix_replay_rows'] = n; immutable_json(directory/'complete.json', r)
    beat('head_complete', head=directory.name, seconds=fit['seconds'], parameters=fit['parameters'])
    return r


def train(resume=False, pilot=False):
    cfg, bcfg, ctx, bid, pid, identity = load(); checked_assets(identity); data = ctx[2]
    refs, ridge, pilots = [], [], []
    for g in groups(bcfg, ctx, bid, pid):
        ti, ids = g['roles']['controller'], g['roles']['readout']
        old = incumbent(g, data, bid, ti); told = incumbent(g, data, bid, ids)
        x, tx = features(g['x'][ti], old), features(g['x'][ids], told)
        env, tenv = g['env'][ti], g['env'][ids]; cv = data['baseline_ade'][ti, 1]
        def ade(p):
            return prior.native_errors(p.astype(float)+data['origin'][ti, None], data['target_eval'][ti], data['valid'][ti], np.ones(len(ti)))[0]
        floor, neural = ade(g['d'][ti]), ade(g['a']['p'][ti])
        for arm in cfg['arms']:
            yy = targets(cv, floor, neural, old, arm=arm, event=g['event'], easy_cut=g['design']['easy_cut'])
            for task, y in zip(('utility', 'risk'), yy):
                hid = dict(experiment=identity, group=g['name'], arm=arm, task=task, event=g['event'], seed=g['seed'],
                    producer=g['fold'], controller=g['controller'], readout=g['roles']['readout_fold'],
                    easy_cut=g['design']['easy_cut'], fitting_ids_sha256=array_hash(ti),
                    feature_train_sha256=array_hash(x), label_sha256=array_hash(y), inference_sha256=array_hash(ids, tx),
                    env_sha256=array_hash(env), incumbent_sha256=array_hash(old, told))
                directory = PRIVATE/'heads'/(g['name']+'_'+arm+'_'+task)
                beat('head_started', head=directory.name, fitting_rows=len(x), pilot=pilot)
                r = fit_one(x, y, data['sites'][ti], cv, env, tx, tenv, ids, directory, hid, cfg, g['seed'], task, resume, pilot)
                if pilot: pilots.append(r)
                else: refs.append(artifact(directory/'complete.json'))
                if not pilot and arm == 'incumbent_reference':
                    directory = PRIVATE/'ridge'/(g['name']+'_'+task)
                    prior.fit_ridge(x, y, cv, data['sites'][ti], tx, tenv, ids, task, hid, directory)
                    ridge.append(artifact(directory/'complete.json'))
            if pilot:
                immutable_json(PRIVATE/'pilot.json', dict(heads=pilots, new_training=True, resumes_inside_budget=True)); return
        beat('training_group_complete', group=g['name'])
    assert len(refs) == 144 and len(ridge) == 72
    immutable_json(PRIVATE/'training_complete.json', dict(identity=identity, heads=refs, ridge=ridge, updates=288000, all_passed=True))


def checked_training(identity):
    r = json.loads((PRIVATE/'training_complete.json').read_text())
    assert r['identity'] == identity and len(r['heads']) == 144 and len(r['ridge']) == 72 and r['all_passed']
    for ref in r['heads']+r['ridge']:
        assert artifact(ROOT/ref['path']) == ref
        rr = json.loads((ROOT/ref['path']).read_text()); assert rr['identity']['experiment'] == identity
        for a in rr['artifacts'].values(): assert artifact(ROOT/a['path']) == a
    return r


def decide():
    cfg, bcfg, ctx, bid, pid, identity = load(); checked_assets(identity); checked_training(identity)
    data = ctx[2]; refs = []
    for g in groups(bcfg, ctx, bid, pid):
        ids = g['roles']['readout']; old = incumbent(g, data, bid, ids)
        moving = np.linalg.norm(data['history'][ids, -1]-data['history'][ids, -2], axis=1) > 0
        out, states = {}, {}
        for arm in (*cfg['arms'], 'ridge_incumbent'):
            vals = []
            for task in ('utility', 'risk'):
                directory = PRIVATE/('ridge' if arm == 'ridge_incumbent' else 'heads')/(g['name']+('_'+task if arm == 'ridge_incumbent' else '_'+arm+'_'+task))
                r = json.loads((directory/'complete.json').read_text()); assert r['identity']['experiment'] == identity
                with np.load(directory/'scores.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], ids); vals.append(z['scores'].copy())
                if arm != 'ridge_incumbent': states[arm, task] = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
            role = 'floor_reference' if arm == 'floor_reference' else 'incumbent_reference'
            out[arm] = choices(*vals, moving, old, arm=role)
            out[arm+'__utility'], out[arm+'__risk'] = vals
            if arm == 'incumbent_reference':
                for name, direction in [('add_only', 'add'), ('remove_only', 'remove')]:
                    out[name] = choices(*vals, moving, old, arm=role, direction=direction)
        for task in ('utility', 'risk'):
            a, b = [states[arm, task] for arm in cfg['arms']]
            np.testing.assert_array_equal(a['draws'], b['draws'])
            for f in ('known', 'weights', 'mean', 'std'):
                np.testing.assert_array_equal(a['preprocess'][f], b['preprocess'][f])
            assert a['preprocess']['cost_scale'] == b['preprocess']['cost_scale']
            assert a['step'] == b['step'] == 2000 and a['settings'] == b['settings'] and a['seed'] == b['seed']
            assert sum(v.numel() for v in a['model'].values()) == sum(v.numel() for v in b['model'].values())
        out['old_stop'] = old
        with np.load(prior.PRIVATE/'decisions'/(g['name']+'.npz'), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); np.testing.assert_array_equal(z['old_stop'], old)
            out['previous_matched'] = z['producer_matched'].copy()
        out['raw_neural'] = np.ones(len(ids), bool)
        path = PRIVATE/'decisions'/(g['name']+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                for k, v in dict(ids=ids, **out).items(): np.testing.assert_array_equal(z[k], v)
        else: np.savez(path, ids=ids, **out)
        immutable_json(path.with_suffix('.json'), dict(identity=identity, group=g['name'], artifact=artifact(path),
            choices=cfg['policies'], supervised_draws_match=True, feature_statistics_match=True, old_choices_exact=True))
        refs.append(artifact(path.with_suffix('.json'))); beat('decisions_frozen', group=g['name'])
    immutable_json(PRIVATE/'decisions_complete.json', dict(identity=identity, groups=refs, training=artifact(PRIVATE/'training_complete.json'), all_passed=True))


def ensure_frozen():
    r = json.loads((PRIVATE/'decisions_complete.json').read_text()); assert r['all_passed'] and len(r['groups']) == 36
    for f, sha in r['identity']['bindings'].items(): assert digest(ROOT/f) == sha
    checked_assets(r['identity']); checked_training(r['identity']); assert artifact(ROOT/r['training']['path']) == r['training']
    for ref in r['groups']:
        assert artifact(ROOT/ref['path']) == ref
        rr = json.loads((ROOT/ref['path']).read_text()); assert rr['identity'] == r['identity']
        assert artifact(ROOT/rr['artifact']['path']) == rr['artifact']


def evaluate(resume=False):
    ensure_frozen(); cfg, bcfg, ctx, bid, pid, identity = load(); data = ctx[2]; refs = []
    for g in groups(bcfg, ctx, bid, pid):
        path = PRIVATE/'evaluation'/(g['name']+'.json')
        if path.exists():
            if not resume: raise ValueError('Existing readout requires resume')
            r = json.loads(path.read_text()); assert r['verified'] and r['identity'] == identity
            refs.append(artifact(path)); continue
        ids = g['roles']['readout']; sites = data['sites'][ids]; roster = identity['rosters'][g['roles']['readout_fold']]
        assert sorted(set(sites)) == roster
        with np.load(PRIVATE/'decisions'/(g['name']+'.npz'), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); arr = {k: z[k].copy() for k in z.files if k != 'ids'}
        moving = np.linalg.norm(data['history'][ids, -1]-data['history'][ids, -2], axis=1) > 0
        for name, arm, direction in [('floor_reference', 'floor_reference', 'both'), ('incumbent_reference', 'incumbent_reference', 'both'),
                ('add_only', 'incumbent_reference', 'add'), ('remove_only', 'incumbent_reference', 'remove'), ('ridge_incumbent', 'ridge_incumbent', 'both')]:
            np.testing.assert_array_equal(arr[name], replay(arr[arm+'__utility'], arr[arm+'__risk'], moving, arr['old_stop'],
                arm='floor_reference' if arm == 'floor_reference' else 'incumbent_reference', direction=direction))
        np.testing.assert_array_equal(arr['old_stop'], incumbent(g, data, bid, ids)); assert arr['raw_neural'].all()
        checks, coords = 0, 0
        def errors(p):
            nonlocal coords
            absolute = p.astype(float)+data['origin'][ids, None]
            native = prior.native_errors(absolute, data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
            other = base.cross.independent.coordinate_errors(absolute, data['target_eval'][ids], data['valid'][ids])
            for a, b in zip(native, other): base.cross.independent.close(a, b); coords += 1
            return native
        def metric(p, ref, mask):
            nonlocal checks
            r = prior.paired_scene_metrics(p[mask], ref[mask], sites[mask], expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel', bootstrap_resamples=3000, seed=39271)
            base.cross.independent.check_metric(p, ref, sites, mask, r, cfg); checks += 1
            return r
        da, df = errors(g['d'][ids]); na, nf = errors(g['a']['p'][ids]); cv = data['baseline_ade'][ids, 1]
        masks = dict(all=np.ones(len(ids), bool), easy=(cv > 0) & (cv <= g['design']['easy_cut']),
            hard=cv >= g['design']['hard_cut'], complete=data['valid'][ids].all(1))
        selected = {p: np.where(arr[p], na, da) for p in cfg['policies']}
        sf = {p: np.where(arr[p], nf, df) for p in cfg['policies']}
        views, known = {}, np.isfinite(cv)
        for p in cfg['policies']:
            use, err, old = arr[p], selected[p], arr['old_stop']
            views[p] = dict(ADE_vs_floor4={s: metric(err, da, m) for s, m in masks.items()},
                ADE_vs_old_stop4={s: metric(err, selected['old_stop'], m) for s, m in masks.items()},
                FDE_vs_floor4=metric(sf[p], df, masks['all']), easy_vs_CV=metric(err, cv, masks['easy']),
                switch_rate=float(use.mean()), switched_rows=int(use.sum()), override_rate=float((use != old).mean()),
                added_rows=int((use & ~old).sum()), removed_rows=int((~use & old).sum()),
                unknown_ADE_switches=int((use & ~known).sum()), zero_CV=dict(rows=int((cv == 0).sum()), harmed_rows=int(((cv == 0) & (err > 0)).sum())))
        for p in (*cfg['arms'], 'ridge_incumbent'):
            role = 'floor_reference' if p == 'floor_reference' else 'incumbent_reference'
            uy, ry = targets(cv, da, na, arr['old_stop'], arm=role, event=g['event'], easy_cut=g['design']['easy_cut'])
            selected_gate = arr[p] if role == 'floor_reference' else arr[p] != arr['old_stop']
            reliability = {}
            for site in roster:
                reliability[site] = {}
                for label, mask in [('population', known & (sites == site)), ('selected', known & (sites == site) & selected_gate)]:
                    den = float(ry[mask, 0].sum()); pred = float(arr[p+'__risk'][mask, 0].sum())
                    reliability[site][label] = dict(rows=int(mask.sum()),
                        utility_mae=np.abs(arr[p+'__utility'][mask]-uy[mask]).mean(0).tolist() if mask.any() else None,
                        risk_mae=np.abs(arr[p+'__risk'][mask]-ry[mask]).mean(0).tolist() if mask.any() else None,
                        realized_harm_ratio=float(ry[mask, 1].sum()/den) if den > 0 else None,
                        predicted_harm_ratio=float(arr[p+'__risk'][mask, 1].sum()/pred) if pred > 0 else None)
            views[p]['reliability'] = reliability
        contrasts = {p: dict(ADE={s: metric(selected['incumbent_reference'], selected[p], m) for s, m in masks.items()},
            FDE=metric(sf['incumbent_reference'], sf[p], masks['all'])) for p in ('floor_reference', 'ridge_incumbent', 'old_stop', 'previous_matched')}
        from scripts.diagnose_m3w_european_fixed_producer_roles import switch_accounting
        accounting = {}
        for p in ('floor_reference', 'incumbent_reference', 'add_only', 'remove_only'):
            accounting[p] = {}
            for s, mask in masks.items():
                accounting[p][s] = {}
                for site in roster:
                    use = mask & known & (sites == site)
                    if not use.any():
                        accounting[p][s][site] = dict(status='no_supported_labels'); continue
                    r = switch_accounting(na[use], da[use], arr[p][use], arr['old_stop'][use])
                    expected = views[p]['ADE_vs_old_stop4'][s]['by_scene'][site]
                    np.testing.assert_allclose([r['new_error_mean'], r['old_error_mean']],
                        [expected['model_error'], expected['reference_error']], rtol=1e-10, atol=1e-10)
                    accounting[p][s][site] = r
        assert checks == 100 and coords == 4
        r = dict(identity=identity, group=g['name'], producer=g['fold'], controller=g['controller'], readout=g['roles']['readout_fold'],
            seed=g['seed'], event=g['event'], views=views, contrasts=contrasts, accounting=accounting, verified=True,
            saved_decisions_verified=8, coordinate_arrays_verified=coords, metric_reductions_verified=checks,
            result_source='fresh_run_new_heads_readout_cached_verified_four_source_forecasts')
        immutable_json(path, r); refs.append(artifact(path)); beat('group_evaluated', group=g['name'], metrics=checks)
    assert len(refs) == 36
    immutable_json(PRIVATE/'evaluation_complete.json', dict(identity=identity, groups=refs, all_passed=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=('prepare', 'pilot', 'train', 'decide', 'evaluate'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); beat('phase_started', phase=args.phase)
        if args.phase == 'prepare': prepare()
        elif args.phase in ('pilot', 'train'): train(args.resume, args.phase == 'pilot')
        elif args.phase == 'decide': decide()
        else: evaluate(args.resume)
        beat('phase_complete', phase=args.phase)


if __name__ == '__main__': main()
