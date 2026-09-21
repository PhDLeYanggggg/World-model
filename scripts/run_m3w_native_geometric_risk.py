"""Registered matched feature/loss factorial for exact-reference geometric risk."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_protected_risk import load as parent_load, training_data
from scripts.run_m3w_native_forecast import array_hash, immutable_json, json_write, assert_current
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_conditional_support import rollout_distance, verify_cost_geometry
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_protected_risk_eval import event_quality
from src.evaluation.m3w_native_matched_coverage import top_count, paired_scene_contrast
from src.world_model.m3w_native_gain_harm import preprocess
from src.world_model.m3w_native_protected_risk import risk_features
from src.world_model.m3w_native_geometric_risk import ARMS, extend_features, zero_targets, build_head, fit, predict
import numpy as np
import torch

CONFIG = 'configs/m3w_native_geometric_risk_v1.json'
CODE = ('scripts/run_m3w_native_geometric_risk.py', 'src/world_model/m3w_native_geometric_risk.py',
        'tests/test_m3w_native_geometric_risk.py')


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    old_reg, parent, data, views, gain, old_id = parent_load()
    path = ROOT/old_reg['reports']/'analysis.json'
    previous = json.loads(path.read_text())
    checked = json.loads((ROOT/old_reg['reports']/'independent_verification.json').read_text())
    if (file_digest(path) != reg['parent_analysis_sha256'] or previous['identity'] != old_id
            or not checked['all_checks_passed'] or checked['analysis_sha256'] != file_digest(path)
            or reg['arms'] != list(ARMS) or reg['training']['steps'] != 3000 or reg['risk_score_cut'] != .01
            or any(reg[k] for k in ('threshold_search', 'model_selection', 'risk_calibration', 'closed_role_readout', 'deployment'))):
        raise ValueError('Fixed verified source factorial required')
    bindings = dict(old_id['source_bindings'])
    for path in (CONFIG, reg['registration'], *CODE, old_reg['reports']+'/analysis.json',
                 old_reg['reports']+'/verification.json', old_reg['reports']+'/independent_verification.json',
                 'outputs/publication_readiness_2026_09/native_conditional_support_v1/analysis.json',
                 'src/evaluation/m3w_native_conditional_support.py'):
        bindings[path] = file_digest(ROOT/path)
    for r in previous['score_archives']:
        if file_digest(ROOT/r['path']) != r['sha256']:
            raise ValueError('Changed prior risk scores')
        bindings[r['path']] = r['sha256']
    return reg, parent, data, views, gain, previous, dict(source_bindings=bindings, parent_identity=old_id,
        scope=reg['scope'], torch=torch.__version__, numpy=np.__version__,
        runtime=dict(torch_threads=4, interop_threads=1, num_workers=0))


def train_view(meta, data, arm):
    ids, x, same, _ = training_data(meta, data)
    with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids']); pred = z['prediction'].copy()
    with np.load(ROOT/meta['targets_path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids'])
        cv, harm, benefit = z['baseline_ade'].copy(), z['harm'].copy(), z['benefit'].copy()
    d = rollout_distance(pred, data['geometry'][ids, 332:356].reshape(-1, 12, 2), data['scale'][ids])
    full = data['valid'][ids].all(1)
    verify_cost_geometry(d, cv, harm, benefit, full)
    x = extend_features(x, data['geometry'][ids], arm)
    y = zero_targets(cv, full)
    pr = preprocess(x, np.column_stack((y, y*d)), np.where(full, cv, np.nan), data['sites'][ids], meta['outer_site'])
    return ids, x, y, d, pr


def trial_identity(identity, meta, ids, x, y, d, pr):
    return dict(identity=identity, outer_site=meta['outer_site'], seed=meta['seed'],
        ids_sha256=array_hash(ids), features_sha256=array_hash(x, d), target_sha256=array_hash(y),
        preprocessing_sha256=array_hash(pr['mean'], pr['std'], pr['known'], pr['weights'], pr['constant']),
        cost_scale=pr['cost_scale'], training_sites=pr['training_sites'], training_producers=meta['training_producers'])


def receipt(path, ti, arm, steps):
    r = json.loads(path.read_text())
    if (r['identity'] != ti or r['arm'] != arm or not r['fit']['complete'] or r['fit']['step'] != steps
            or file_digest(ROOT/r['checkpoint']) != r['checkpoint_sha256']):
        raise ValueError('Changed/incomplete geometric head endpoint')
    return r


def train(reg, data, views, identity, args, beat):
    if args.view and args.view not in views:
        raise ValueError('Unregistered source view')
    for key, meta in views.items():
        if args.view and args.view != key:
            continue
        for arm in ARMS:
            if args.arm and args.arm != arm:
                continue
            ids, x, y, d, pr = train_view(meta, data, arm)
            ti = trial_identity(identity, meta, ids, x, y, d, pr)
            directory = ROOT/reg['output']/'trials'/key/arm; rp = directory/'complete.json'
            if rp.exists():
                receipt(rp, ti, arm, reg['training']['steps']); beat(state='cached_verified', view=key, arm=arm); continue
            beat(state='fitting_geometric_risk', view=key, arm=arm, supported_rows=int(pr['known'].sum()))
            result = fit(x, y, d, data['sites'][ids], pr, arm=arm, seed=meta['seed'], settings=reg['training'],
                identity=ti, directory=directory, resume=args.resume, stop_at=args.stop_at,
                heartbeat=lambda **v:beat(view=key, arm=arm, **v))
            if not result['complete']:
                beat(state='pilot_complete', fit=result); return
            cp = directory/'checkpoint.pt'; assert_current(identity)
            immutable_json(rp, dict(identity=ti, arm=arm, fit=result, checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=file_digest(cp)))
    beat(state='registered_training_complete')


def evaluate(reg, parent, data, views, gain, previous, identity, verify, beat):
    root, public = ROOT/reg['output'], ROOT/reg['reports']
    records = {}
    for key in views:
        for arm in ARMS:
            rp = root/'trials'/key/arm/'complete.json'; r = json.loads(rp.read_text())
            if r['identity']['identity'] != identity:
                raise ValueError('Mixed endpoint lineage')
            records[key, arm] = receipt(rp, r['identity'], arm, reg['training']['steps'])
    n = len(data['sites']); full = data['valid'].all(1)
    cv, cf = native_errors(data['geometry'][:, 332:356].reshape(-1, 12, 2), data['target'], data['valid'], data['scale'])
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    quality, capacities, scores, arrays = [], [], [], {}
    sample_checks = 0
    for key, meta in views.items():
        site, seed = meta['outer_site'], meta['seed']
        with np.load(ROOT/meta['outer_prediction']['prediction']['path'], allow_pickle=False) as z:
            ids, pred = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == site))
        x0, same = risk_features(data['geometry'][ids], pred, data['scale'][ids])
        d = rollout_distance(pred, data['geometry'][ids, 332:356].reshape(-1, 12, 2), data['scale'][ids])
        ne, nf = native_errors(pred, data['target'][ids], data['valid'][ids], data['scale'][ids])
        y = zero_targets(cv[ids], full[ids]); true_cost = np.where(full[ids], np.maximum(ne-cv[ids], 0)*(cv[ids] == 0), np.nan)
        probabilities, first_draw = {}, None
        for arm in ARMS:
            ti_ids, tx, ty, td, pr = train_view(meta, data, arm)
            ti = trial_identity(identity, meta, ti_ids, tx, ty, td, pr)
            r = receipt(root/'trials'/key/arm/'complete.json', ti, arm, reg['training']['steps'])
            cp = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
            assert cp['identity'] == ti and cp['step'] == reg['training']['steps'] and cp['settings'] == reg['training']
            for k in ('mean', 'std', 'weights', 'known', 'constant'):
                np.testing.assert_array_equal(cp['preprocess'][k], pr[k])
            assert cp['draws'][~pr['known']].sum() == 0 and cp['draws'].sum() == reg['training']['steps']*reg['training']['batch_size']
            if first_draw is None:
                first_draw = cp['draws']
            else:
                np.testing.assert_array_equal(first_draw, cp['draws']); sample_checks += 1
            model = build_head(seed); model.load_state_dict(cp['model'])
            p = predict(model, extend_features(x0, data['geometry'][ids], arm), pr)
            probabilities[arm] = p
            f = full[ids]; cost = p*d
            quality.append(dict(view=key, arm=arm, zero_reference_event=event_quality(p, y, f),
                protected_harm_event=event_quality(np.where(same, 0., p), np.where(same, 0., y), f),
                native_protected_cost_mse=float(np.mean((cost[f]-true_cost[f])**2)),
                probability_is_calibrated=False))
        path = root/'scores'/(key+'.npz'); rp = root/'scores'/(key+'.json')
        if verify and not path.exists():
            raise ValueError('Replay may not create missing scores')
        if rp.exists() and file_digest(path) != json.loads(rp.read_text())['sha256']:
            raise ValueError('Changed scores')
        write_arrays(path, dict(ids=ids, distance=d, **probabilities))
        sr = dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path), ids_sha256=array_hash(ids))
        immutable_json(rp, sr); scores.append(sr)
        sb = next(r for r in gain['score_archives'] if r['view'] == key)
        with np.load(ROOT/sb['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); m, a = z['mse'].copy(), z['underharm4'].copy()
        rb = next(r for r in previous['score_archives'] if r['view'] == key)
        with np.load(ROOT/rb['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); old_p = z['zero_reference_harm'][:, 0].copy()
        net = m[:, 0]-m[:, 1]; positive = (net > 0) & ~same
        strict = positive & (m[:, 1] <= .1*m[:, 0]); stop = x0[:, -2] == 1
        choices = dict(net_only=positive, mse_strict=strict, stop_mse_strict=strict & ~stop,
            previous_protected_net=positive & (old_p <= .01), previous_protected_strict=strict & (old_p <= .01))
        pools = dict(mse=strict, stop=strict & ~stop, previous=strict & (old_p <= .01))
        for arm, p in probabilities.items():
            choices[arm+'_net'] = positive & (p <= .01)
            choices[arm+'_strict'] = strict & (p <= .01)
            pools[arm] = choices[arm+'_strict']
        anchor = (a[:, 0] > a[:, 1]) & (a[:, 1] <= .1*a[:, 0]) & ~same
        common = min(int(anchor.sum()), *(int(v.sum()) for v in pools.values()))
        choices.update({k+'_matched':top_count(net, v, ids, common) for k,v in pools.items()})
        capacities.append(dict(view=key, requested_count=int(anchor.sum()), common_count=common,
            capacity={k:int(v.sum()) for k,v in pools.items()}, selected_ids_sha256={k:array_hash(ids[v]) for k,v in choices.items()}))
        rr = next(r for r in gain['training'] if r['view'] == key and r['arm'] == 'ridge')
        pr = torch.load(ROOT/rr['checkpoint'], map_location='cpu', weights_only=False)['preprocess']
        masks['hard'][ids] = cv[ids] >= pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= pr['positive_easy_cut'])
        for name, use in choices.items():
            storage = arrays.setdefault((name, seed), dict(ade=np.full(n, np.nan), fde=np.full(n, np.nan), use=np.zeros(n, bool), seen=np.zeros(n, bool)))
            if storage['seen'][ids].any():
                raise ValueError('Duplicate query')
            storage['ade'][ids], storage['fde'][ids] = np.where(use, ne, cv[ids]), np.where(use, nf, cf[ids])
            storage['use'][ids], storage['seen'][ids] = use, True
        beat(state='held_source_readout', view=key, common_count=common)
    def metric(v, ref, mask=None):
        if mask is None:
            mask = np.ones(n, bool)
        return paired_scene_metrics(v[mask], ref[mask], data['sites'][mask], expected_scenes=parent['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=3000, seed=38113)
    summary = {}
    for name in sorted({k[0] for k in arrays}):
        seeds = {}
        for seed in parent['seeds']:
            r = arrays[name, seed]; assert r['seen'].all()
            seeds[str(seed)] = dict(ADE=metric(r['ade'], cv), FDE=metric(r['fde'], cf),
                subsets={k:metric(r['ade'], cv, v) for k,v in masks.items()},
                intervention_rate=float(r['use'].mean()), zero_CV_harmed_rows=int((r['ade'][masks['zero_CV']] > 0).sum()),
                zero_CV_max_absolute_harm=float(r['ade'][masks['zero_CV']].max()),
                unknown_ADE_selected_rows=int((r['use'] & ~np.isfinite(cv)).sum()),
                incomplete_risk_selected_rows=int((r['use'] & ~full).sum()))
        mean = np.mean([arrays[name, s]['ade'] for s in parent['seeds']], axis=0)
        summary[name] = dict(seeds=seeds, ADE=metric(mean, cv), subsets={k:metric(mean, cv, v) for k,v in masks.items()})
    contrasts = {}
    for left, right in (('base_geometric', 'base_event'), ('kinematic_event', 'base_event'),
                        ('kinematic_geometric', 'base_geometric'), ('kinematic_geometric', 'kinematic_event')):
        for rule in ('strict', 'matched'):
            aa, bb = (summary[k+'_'+rule]['ADE']['by_scene'] for k in (left, right))
            contrasts[left+'_minus_'+right+'_'+rule] = paired_scene_contrast(
                [aa[s]['gain_percent'] for s in parent['sites']], [bb[s]['gain_percent'] for s in parent['sites']])
    result = dict(identity=identity, result_source='fresh_run_geometric_risk_training_cached_verified_predictors_and_gain_heads',
        training=[dict(view=k, arm=a, **{f:r[f] for f in ('fit','checkpoint','checkpoint_sha256')}) for (k,a),r in records.items()],
        optimizer_updates=sum(r['fit']['step'] for r in records.values()), summed_fit_seconds=sum(r['fit']['seconds'] for r in records.values()),
        paired_sampler_checks=sample_checks, quality=quality, summary=summary, contrasts=contrasts,
        score_archives=scores, capacities=capacities, independent_confirmation=False, deployment=False,
        risk_calibrated=False, threshold_search=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity); immutable_json(public/'analysis.json', result)
    if verify:
        immutable_json(public/'verification.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
            checked_heads=len(records), probability_rows_replayed=n*len(parent['seeds'])*len(ARMS),
            paired_sampler_checks=sample_checks, all_checks_passed=True, new_training=False))
    beat(state='verified' if verify else 'evaluated', heads=len(records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('audit-only', 'resume', 'evaluate', 'verify'):
        p.add_argument('--'+flag, action='store_true')
    p.add_argument('--view'); p.add_argument('--arm', choices=ARMS); p.add_argument('--stop-at', type=int)
    args = p.parse_args()
    if args.stop_at is not None and (not args.view or not args.arm or args.evaluate or args.verify):
        raise ValueError('Pilot must identify one fixed training view/arm')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, parent, data, views, gain, previous, identity = load()
    root = ROOT/reg['output']; root.mkdir(parents=True, exist_ok=True)
    lock = (root/'execution.lock').open('a'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    def beat(**v):
        e = dict(pid=os.getpid(), timestamp_unix=time.time(), **v)
        json_write(root/'heartbeat.json', e)
        with (root/'events.jsonl').open('a') as f:
            f.write(json.dumps(e)+'\n')
        print(json.dumps(e), flush=True)
    if args.audit_only:
        beat(state='preflight_pass', bindings=len(identity['source_bindings']), views=len(views)); return
    immutable_json(root/'identity.json', identity)
    if args.evaluate or args.verify:
        evaluate(reg, parent, data, views, gain, previous, identity, args.verify, beat)
    else:
        train(reg, data, views, identity, args, beat)


if __name__ == '__main__':
    main()
