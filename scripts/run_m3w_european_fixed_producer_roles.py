"""Fixed four-source forecasts with disjoint producer/controller/readout roles."""
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
from scripts import run_m3w_european_floor_relative as base
from scripts import run_m3w_european_producer_conditioned as previous
from scripts import run_m3w_european_causal_abstention as stopped
from src.world_model.m3w_fixed_producer_roles import role_rosters, role_indices, bounded_ridge_scores, label_free_replay, forecaster_baseline
from src.world_model.m3w_floor_relative import matched_features, relative_targets, fixed_rank_scale
from src.world_model.m3w_european_source_forecast import SourceForecaster
from src.world_model.m3w_native_forecast import predict as forecast_predict
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_fixed_producer_roles_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_fixed_producer_roles_v1'
CONFIG = 'configs/m3w_european_fixed_producer_roles_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_fixed_producer_roles.py', 'src/world_model/m3w_fixed_producer_roles.py',
    'tests/test_m3w_fixed_producer_roles.py', 'tests/test_m3w_fixed_producer_roles_protocol.py',
    'outputs/publication_readiness_2026_09/european_fixed_producer_roles_v1/registration.md')
digest, artifact, immutable_json, array_hash = base.digest, base.artifact, base.immutable_json, base.array_hash


def beat(state, **kw):
    r = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    base.cross.json_write(PRIVATE/'heartbeat.json', r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r), flush=True)


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    expected = dict(mode='fitting', seeds=[17, 29, 43], events=['all', 'easy'], arms=['producer_matched', 'oof_control'],
        policies=['producer_matched', 'oof_control', 'ridge', 'old_stop', 'raw_neural'], groups=36,
        new_neural_heads=144, neural_updates=288000, ridge_fits=72, views=180, risk_budget=.02,
        ridge_alpha=.01, rank_weight=1., rank_epsilon=1e-6, fixed_scale_batches=40,
        bootstrap_resamples=3000, bootstrap_seed=39271, replay_rows=4096)
    for k, v in expected.items():
        if cfg[k] != v: raise ValueError('Changed registered comparison: '+k)
    assert not any(cfg[k] for k in ('new_forecaster_training', 'threshold_refit', 'calibration_refit',
        'reserved_roles_opened', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    assert digest(previous.PUBLIC/'summary_metrics.json') == cfg['prior_summary_sha256']
    done = json.loads((previous.PUBLIC/'completion_checks.json').read_text()); assert done['all_passed']
    for f, sha in done['artifact_hashes'].items():
        if digest(previous.PUBLIC/f) != sha: raise ValueError('Previous evidence changed')
    stopped.ensure_both_frozen()
    bcfg, ctx, _, bid = base.load('fitting'); assert cfg['head_training'] == bcfg['head_training']
    pid = ctx[3]['parent_identity']['geometric_identity']['old_identity']['producer_identity']
    rosters = role_rosters(pid['halves'])
    identity = dict(bindings={f: digest(ROOT/f) for f in FILES}, parent=bid, rosters=rosters,
        previous_completion=artifact(previous.PUBLIC/'completion_checks.json'))
    immutable_json(PRIVATE/'identity.json', identity)
    return cfg, bcfg, ctx, bid, identity


def groups(bcfg, ctx, bid, identity):
    for name, fold, seed, event, design, a, d, x, env, bits, provenance in base.groups(bcfg, ctx, bid):
        for controller in range(3):
            if controller == fold: continue
            roles = role_indices(ctx[2]['sites'], identity['rosters'], fold, controller)
            np.testing.assert_array_equal(roles['producer'], design['train_ids'])
            assert set(a['lineage']['final_producer']['training_sites']) == set(identity['rosters'][fold])
            assert not set(identity['rosters'][fold]) & set(ctx[2]['sites'][roles['readout']])
            yield dict(name=name+f'_controller{controller}', original=name, fold=fold, controller=controller,
                seed=seed, event=event, design=design, a=a, d=d, x=x, env=env, bits=bits,
                provenance=provenance, roles=roles)


def old_scores(g, data, bid):
    ids = g['roles']['readout']; values = []
    for task in ('utility', 'risk'):
        r = json.loads((base.PRIVATE/'fitting/heads'/(g['original']+'_floor_'+task)/'complete.json').read_text())
        assert r['identity']['experiment'] == bid
        pred, _ = previous.restored(r); values.append(pred(g['x'][ids], g['env'][ids]))
    moving = np.linalg.norm(data['history'][ids, -1]-data['history'][ids, -2], axis=1) > 0
    choice = previous.safe_choice(*values, moving)
    with np.load(stopped.PRIVATE/'fitting/decisions'/(g['original']+'.npz'), allow_pickle=False) as z:
        where = np.searchsorted(z['ids'], ids); np.testing.assert_array_equal(z['ids'][where], ids)
        np.testing.assert_array_equal(choice, z['floor_both__stop'][where])
    return values, choice


def prepare():
    cfg, bcfg, ctx, bid, identity = load(); data = ctx[2]; replayed = set(); rows = []
    fcfg = json.loads((ROOT/'configs/m3w_european_source_forecast_v1.json').read_text())
    for g in groups(bcfg, ctx, bid, identity):
        key = (g['fold'], g['seed']); ref = g['a']['lineage']['final_producer']; ids = g['roles']['readout']
        for item in ('checkpoint', 'prediction'): assert artifact(ROOT/ref[item]['path']) == ref[item]
        if key not in replayed:
            state = torch.load(ROOT/ref['checkpoint']['path'], map_location='cpu', weights_only=False)
            selected = forecaster_baseline(state, identity['rosters'][g['fold']], g['fold'], g['seed'])
            model = SourceForecaster(fcfg['architecture'], selected)
            model.load_state_dict(state['model']); prefix = ids[:cfg['replay_rows']]
            np.testing.assert_array_equal(forecast_predict(model, data, prefix, 128), g['a']['p'][prefix])
            replayed.add(key)
        old_scores(g, data, bid)
        rows.append(dict(group=g['name'], producer=g['fold'], controller=g['controller'], readout=g['roles']['readout_fold'],
            role_rows={k: len(g['roles'][k]) for k in ('producer', 'controller', 'readout')},
            role_hashes={k: array_hash(g['roles'][k]) for k in ('producer', 'controller', 'readout')},
            easy_cut=g['design']['easy_cut'], hard_cut=g['design']['hard_cut']))
        beat('assets_verified', group=g['name'])
    assert len(rows) == 36 and len(replayed) == 9
    immutable_json(PRIVATE/'assets_complete.json', dict(identity=identity, groups=rows, source_exclusion_pass=True,
        cached_neural_prefix_replays=9, cached_old_head_replays=72, old_choices_exact=36, new_readout=False, all_passed=True))


def checked_assets(identity):
    r = json.loads((PRIVATE/'assets_complete.json').read_text())
    assert r['identity'] == identity and r['all_passed'] and len(r['groups']) == 36


def controller_data(g, arm, bcfg, ctx, bid):
    data = ctx[2]; ids = g['roles']['controller']
    if arm == 'producer_matched':
        return g['x'][ids], g['env'][ids], g['d'][ids], g['a']['p'][ids], dict(neural=g['a']['lineage'], floor=g['provenance'])
    fold, seed = g['controller'], g['seed']
    design = next(d for _, candidate, f, s, d in base.cross.parent.jobs(ctx[1], data, ctx[3]['parent_identity'])
                  if (candidate, f, s) == ('neural', fold, seed))
    np.testing.assert_array_equal(ids, design['train_ids'])
    a = base.cross.parent.assemble('neural', fold, seed, design, data, ctx[3]['parent_identity'])
    bits, provenance = base.floor_bits(bcfg, ctx, bid, fold, seed, g['event'], design)
    damping = base.baseline_numpy(data['history'][ids], 3)-data['origin'][ids, None]
    d = np.where(bits[ids, None, None], damping, a['b'][ids])
    x, env = matched_features(data['geometry'][ids], a['b'][ids], d, a['p'][ids], bits[ids])
    return x, env, d, a['p'][ids], dict(neural=a['lineage'], floor=provenance)


def fit_one(x, y, sites, cv, env, tx, tenv, ids, *, directory, hid, cfg, seed, task, resume, pilot):
    if (directory/'complete.json').exists(): return base.checked(directory, hid)
    if shutil.disk_usage(PRIVATE).free < 10*1024**3: raise OSError('Below10GiB; preserve checkpoints')
    pr = base.ordinary.preprocess(x, y, cv, sites, '__excluded_outer__')
    kwargs = dict(seed=seed, settings=cfg['head_training'], identity=hid, directory=directory, resume=resume,
        stop_at=100 if pilot else None, heartbeat=lambda **kw: beat(head=directory.name, **kw))
    if task == 'utility':
        model, fit = base.geometric.fit(x, y, sites, env, pr, task=task, **kwargs)
        pred = lambda xx, dd: base.geometric.predict(model, xx, dd, pr); kind = 'bounded_utility'
    else:
        fixed = fixed_rank_scale(y, sites, pr, seed=seed, batch_size=256, batches=40)
        model, fit = base.ranked.fit(x, y, sites, env, pr, rank_weight=1., epsilon=1e-6, fixed_denominator=fixed, **kwargs)
        pred = lambda xx, dd: base.hurdle.predict(model, xx, dd, pr); kind = 'ranked_risk'
    if pilot: return dict(identity=hid, fit=fit, checkpoint=artifact(directory/'checkpoint.pt'))
    scores = pred(tx, tenv); path = directory/'scores.npz'; temp = path.with_suffix('.tmp.npz')
    np.savez(temp, ids=ids, scores=scores); os.replace(temp, path)
    r = dict(identity=hid, fit=fit, kind=kind, replay_exact=True,
        artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'), scores=artifact(path)))
    fresh, _ = previous.restored(r); n = min(len(ids), cfg['replay_rows'])
    np.testing.assert_array_equal(fresh(tx[:n], tenv[:n]), scores[:n])
    assert fit['unknown_rows_sampled'] == 0
    r['prefix_replay_rows'] = n; immutable_json(directory/'complete.json', r)
    beat('head_complete', head=directory.name, seconds=fit['seconds'], parameters=fit['parameters'])
    return r


def fit_ridge(x, y, cv, sites, tx, tenv, ids, task, hid, directory):
    if (directory/'complete.json').exists():
        r = json.loads((directory/'complete.json').read_text()); assert r['identity'] == hid
        for ref in r['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
        return r
    pr = base.ordinary.preprocess(x, y, cv, sites, '__excluded_outer__')
    head = base.ordinary.fit_ridge(x, y, pr, alpha=.01)
    scores = bounded_ridge_scores(base.ordinary.predict_ridge(head, tx, np.zeros(len(tx), bool), pr), tenv, task)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(dict(identity=hid, preprocess=pr, head=head), directory/'ridge.pt')
    np.savez(directory/'scores.npz', ids=ids, scores=scores)
    state = torch.load(directory/'ridge.pt', map_location='cpu', weights_only=False)
    replay = bounded_ridge_scores(base.ordinary.predict_ridge(state['head'], tx, np.zeros(len(tx), bool), state['preprocess']), tenv, task)
    np.testing.assert_array_equal(scores, replay)
    r = dict(identity=hid, full_replay_exact=True, artifacts=dict(model=artifact(directory/'ridge.pt'), scores=artifact(directory/'scores.npz')))
    immutable_json(directory/'complete.json', r); return r


def train(resume=False, pilot=False):
    cfg, bcfg, ctx, bid, identity = load(); checked_assets(identity); data = ctx[2]; refs = []; ridge = []; pilots = []
    for g in groups(bcfg, ctx, bid, identity):
        train_ids, ids = g['roles']['controller'], g['roles']['readout']
        tx, tenv = g['x'][ids], g['env'][ids]; cv = data['baseline_ade'][train_ids, 1]
        for arm in cfg['arms']:
            x, env, d, n, lineage = controller_data(g, arm, bcfg, ctx, bid)
            def ade(p):
                return native_errors(p.astype(float)+data['origin'][train_ids, None], data['target_eval'][train_ids],
                    data['valid'][train_ids], np.ones(len(train_ids)))[0]
            targets = relative_targets(cv, ade(d), ade(n), reference='floor', event=g['event'], easy_cut=g['design']['easy_cut'])
            for task, y in zip(('utility', 'risk'), targets):
                hid = dict(experiment=identity, group=g['name'], arm=arm, task=task, event=g['event'], seed=g['seed'],
                    producer=g['fold'], controller=g['controller'], readout=g['roles']['readout_fold'],
                    easy_cut=g['design']['easy_cut'], fitting_ids_sha256=array_hash(train_ids),
                    feature_train_sha256=array_hash(x), label_sha256=array_hash(y), inference_sha256=array_hash(ids, tx),
                    env_sha256=array_hash(env), lineage=lineage)
                directory = PRIVATE/'heads'/(g['name']+'_'+arm+'_'+task)
                beat('head_started', head=directory.name, fitting_rows=len(x), pilot=pilot)
                r = fit_one(x, y, data['sites'][train_ids], cv, env, tx, tenv, ids, directory=directory, hid=hid,
                    cfg=cfg, seed=g['seed'], task=task, resume=resume, pilot=pilot)
                if pilot: pilots.append(r)
                else: refs.append(artifact(directory/'complete.json'))
                if not pilot and arm == 'producer_matched':
                    directory = PRIVATE/'ridge'/(g['name']+'_'+task)
                    fit_ridge(x, y, cv, data['sites'][train_ids], tx, tenv, ids, task, hid, directory)
                    ridge.append(artifact(directory/'complete.json'))
            if pilot:
                immutable_json(PRIVATE/'pilot.json', dict(heads=pilots, new_training=True, resumes_inside_budget=True)); return
        beat('training_group_complete', group=g['name'])
    assert len(refs) == 144 and len(ridge) == 72
    immutable_json(PRIVATE/'training_complete.json', dict(identity=identity, heads=refs, ridge=ridge, all_passed=True, updates=288000))


def checked_training(identity):
    r = json.loads((PRIVATE/'training_complete.json').read_text())
    assert r['identity'] == identity and len(r['heads']) == 144 and len(r['ridge']) == 72 and r['all_passed']
    for ref in r['heads']+r['ridge']:
        assert artifact(ROOT/ref['path']) == ref
        rr = json.loads((ROOT/ref['path']).read_text()); assert rr['identity']['experiment'] == identity
        for a in rr['artifacts'].values(): assert artifact(ROOT/a['path']) == a
    return r


def decide():
    cfg, bcfg, ctx, bid, identity = load(); checked_assets(identity); checked_training(identity); data = ctx[2]; refs = []
    for g in groups(bcfg, ctx, bid, identity):
        ids = g['roles']['readout']; moving = np.linalg.norm(data['history'][ids, -1]-data['history'][ids, -2], axis=1) > 0
        out = {}; states = {}
        for arm in (*cfg['arms'], 'ridge'):
            values = []
            for task in ('utility', 'risk'):
                directory = (PRIVATE/'ridge'/(g['name']+'_'+task) if arm == 'ridge' else PRIVATE/'heads'/(g['name']+'_'+arm+'_'+task))
                r = json.loads((directory/'complete.json').read_text()); assert r['identity']['experiment'] == identity
                with np.load(directory/'scores.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], ids); values.append(z['scores'].copy())
                if arm != 'ridge': states[arm, task] = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
            out[arm] = previous.safe_choice(*values, moving)
            out[arm+'__utility'], out[arm+'__risk'] = values
        for task in ('utility', 'risk'):
            a, b = [states[arm, task] for arm in cfg['arms']]
            np.testing.assert_array_equal(a['draws'], b['draws'])
            for f in ('known', 'weights'): np.testing.assert_array_equal(a['preprocess'][f], b['preprocess'][f])
            assert a['preprocess']['cost_scale'] == b['preprocess']['cost_scale']
            assert a['step'] == b['step'] == 2000 and a['settings'] == b['settings'] and a['seed'] == b['seed']
            assert sum(v.numel() for v in a['model'].values()) == sum(v.numel() for v in b['model'].values())
        vals, out['old_stop'] = old_scores(g, data, bid); out['old_stop__utility'], out['old_stop__risk'] = vals
        out['raw_neural'] = np.ones(len(ids), bool)
        path = PRIVATE/'decisions'/(g['name']+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                for k, v in dict(ids=ids, **out).items(): np.testing.assert_array_equal(z[k], v)
        else: np.savez(path, ids=ids, **out)
        immutable_json(path.with_suffix('.json'), dict(identity=identity, group=g['name'], artifact=artifact(path),
            choices=cfg['policies'], supervised_draws_match=True, old_choices_exact=True))
        refs.append(artifact(path.with_suffix('.json'))); beat('decisions_frozen', group=g['name'])
    immutable_json(PRIVATE/'decisions_complete.json', dict(identity=identity, groups=refs,
        training=artifact(PRIVATE/'training_complete.json'), all_passed=True, new_readout=False))


def ensure_frozen():
    r = json.loads((PRIVATE/'decisions_complete.json').read_text()); assert r['all_passed'] and len(r['groups']) == 36
    for f, sha in r['identity']['bindings'].items(): assert digest(ROOT/f) == sha
    checked_assets(r['identity']); checked_training(r['identity']); assert artifact(ROOT/r['training']['path']) == r['training']
    for ref in r['groups']:
        assert artifact(ROOT/ref['path']) == ref
        rec = json.loads((ROOT/ref['path']).read_text()); assert rec['identity'] == r['identity']
        assert artifact(ROOT/rec['artifact']['path']) == rec['artifact']


def evaluate(resume=False):
    ensure_frozen(); cfg, bcfg, ctx, bid, identity = load(); data = ctx[2]; refs = []
    for g in groups(bcfg, ctx, bid, identity):
        path = PRIVATE/'evaluation'/(g['name']+'.json')
        if path.exists():
            if not resume: raise ValueError('Existing readout requires resume')
            r = json.loads(path.read_text()); assert r['verified'] and r['identity'] == identity
            refs.append(artifact(path)); continue
        ids = g['roles']['readout']; sites = data['sites'][ids]; roster = identity['rosters'][g['roles']['readout_fold']]
        assert sorted(set(sites)) == roster
        moving = np.linalg.norm(data['history'][ids, -1]-data['history'][ids, -2], axis=1) > 0
        with np.load(PRIVATE/'decisions'/(g['name']+'.npz'), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); arr = {k: z[k].copy() for k in z.files if k != 'ids'}
        for arm in cfg['policies'][:-1]:
            np.testing.assert_array_equal(arr[arm], label_free_replay((arr[arm+'__utility'], arr[arm+'__risk']), moving))
        assert arr['raw_neural'].all()
        checks = 0; coords = 0
        def errors(p):
            nonlocal coords
            absolute = p.astype(float)+data['origin'][ids, None]
            native = native_errors(absolute, data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
            other = base.cross.independent.coordinate_errors(absolute, data['target_eval'][ids], data['valid'][ids])
            for a, b in zip(native, other): base.cross.independent.close(a, b); coords += 1
            return native
        def metric(p, ref, mask):
            nonlocal checks
            r = paired_scene_metrics(p[mask], ref[mask], sites[mask], expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel', bootstrap_resamples=3000, seed=39271)
            base.cross.independent.check_metric(p, ref, sites, mask, r, cfg); checks += 1
            return r
        da, df = errors(g['d'][ids]); na, nf = errors(g['a']['p'][ids]); cv = data['baseline_ade'][ids, 1]
        masks = dict(all=np.ones(len(ids), bool), easy=(cv > 0) & (cv <= g['design']['easy_cut']),
            hard=cv >= g['design']['hard_cut'], complete=data['valid'][ids].all(1))
        selected = {arm: np.where(arr[arm], na, da) for arm in cfg['policies']}
        sf = {arm: np.where(arr[arm], nf, df) for arm in cfg['policies']}
        uy, ry = relative_targets(cv, da, na, reference='floor', event=g['event'], easy_cut=g['design']['easy_cut'])
        known = np.isfinite(cv); views = {}
        for arm in cfg['policies']:
            use, err = arr[arm], selected[arm]; reliability = {}
            if arm != 'raw_neural':
                up, rp = arr[arm+'__utility'], arr[arm+'__risk']
                for site in roster:
                    reliability[site] = {}
                    for label, mask in [('population', known & (sites == site)), ('selected', known & (sites == site) & use)]:
                        den = float(ry[mask, 0].sum()); pred = float(rp[mask, 0].sum())
                        reliability[site][label] = dict(rows=int(mask.sum()),
                            utility_mae=np.abs(up[mask]-uy[mask]).mean(0).tolist() if mask.any() else None,
                            risk_mae=np.abs(rp[mask]-ry[mask]).mean(0).tolist() if mask.any() else None,
                            realized_harm_ratio=float(ry[mask, 1].sum()/den) if den > 0 else None,
                            predicted_harm_ratio=float(rp[mask, 1].sum()/pred) if pred > 0 else None)
            views[arm] = dict(ADE_vs_floor4={s: metric(err, da, m) for s, m in masks.items()},
                ADE_vs_old_stop4={s: metric(err, selected['old_stop'], m) for s, m in masks.items()},
                FDE_vs_floor4=metric(sf[arm], df, masks['all']), easy_vs_CV=metric(err, cv, masks['easy']),
                switch_rate=float(use.mean()), switched_rows=int(use.sum()), unknown_ADE_switches=int((use & ~known).sum()),
                zero_CV=dict(rows=int((cv == 0).sum()), harmed_rows=int(((cv == 0) & (err > 0)).sum())),
                reliability=reliability)
        contrasts = {arm: dict(ADE={s: metric(selected['producer_matched'], selected[arm], m) for s, m in masks.items()},
            FDE=metric(sf['producer_matched'], sf[arm], masks['all'])) for arm in ('oof_control', 'ridge', 'old_stop')}
        floor_easy = metric(da, cv, masks['easy'])
        assert len(views) == 5 and checks == 66 and coords == 4
        r = dict(identity=identity, group=g['name'], producer=g['fold'], controller=g['controller'], readout=g['roles']['readout_fold'],
            seed=g['seed'], event=g['event'], views=views, contrasts=contrasts, floor_easy_vs_CV=floor_easy, verified=True,
            saved_decisions_verified=5, coordinate_arrays_verified=coords, metric_reductions_verified=checks,
            result_source='fresh_run_new_controller_readout_cached_verified_four_source_forecasts')
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
