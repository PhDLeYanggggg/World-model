"""Matched-capacity source-only event/cost fitting and protected-risk controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_matched_coverage import load as load_parent
from scripts.run_m3w_native_forecast import array_hash, assert_current, immutable_json, json_write
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_native_cost_readout import fixed_interventions
from src.evaluation.m3w_native_protected_risk_eval import risk_decisions, event_quality
from src.world_model.m3w_native_gain_harm import preprocess
from src.world_model.m3w_native_protected_risk import risk_features, risk_targets, fit_head, build_head, predict_head
import numpy as np
import torch

CONFIG = 'configs/m3w_native_protected_risk_v1.json'
CODE = ('scripts/run_m3w_native_protected_risk.py', 'src/world_model/m3w_native_protected_risk.py',
        'src/evaluation/m3w_native_protected_risk_eval.py', 'tests/test_m3w_native_protected_risk.py')


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    matched, parent, data, views, gain, parent_identity = load_parent()
    pa = ROOT/matched['reports']/'analysis.json'
    checked = json.loads((ROOT/matched['reports']/'independent_verification.json').read_text())
    if (file_digest(pa) != reg['parent_matched_analysis_sha256'] or checked['analysis_sha256'] != file_digest(pa)
            or not checked['all_checks_passed'] or reg['arms'] != ['all_harm', 'zero_reference_harm']
            or reg['risk_score_cut'] != .01 or reg['feature_dim'] != 357 or reg['training']['steps'] != 3000
            or any(reg[k] for k in ('model_selection', 'threshold_search', 'risk_calibration', 'closed_role_readout', 'deployment'))):
        raise ValueError('Fixed, verified source-only risk comparison required')
    bindings = dict(parent_identity['source_bindings'])
    for path in (CONFIG, reg['registration'], *CODE, matched['reports']+'/analysis.json',
                 matched['reports']+'/verification.json', matched['reports']+'/independent_verification.json',
                 matched['reports']+'/risk_target_support.json'):
        bindings[path] = file_digest(ROOT/path)
    return reg, parent, data, views, gain, dict(source_bindings=bindings, parent_identity=parent_identity,
        scope=reg['scope'], torch=torch.__version__, numpy=np.__version__,
        runtime=dict(torch_threads=4, interop_threads=1, num_workers=0))


def training_data(meta, data):
    with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
        if set(z.files) != {'ids', 'prediction'}:
            raise ValueError('No labels permitted in the training input archive')
        ids, prediction = z['ids'].copy(), z['prediction'].copy()
    np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] != meta['outer_site']))
    with np.load(ROOT/meta['targets_path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids'])
        cv, harm = z['baseline_ade'].copy(), z['harm'].copy()
    x, same = risk_features(data['geometry'][ids], prediction, data['scale'][ids])
    full = data['valid'][ids].all(1)
    output = {}
    for arm in ('all_harm', 'zero_reference_harm'):
        y = risk_targets(cv, harm, full, arm)
        pr = preprocess(x, y, np.where(full, cv, np.nan), data['sites'][ids], meta['outer_site'])
        output[arm] = y, pr
    for field in ('mean', 'std', 'known', 'weights'):
        np.testing.assert_array_equal(output['all_harm'][1][field], output['zero_reference_harm'][1][field])
    assert output['all_harm'][1]['cost_scale'] == output['zero_reference_harm'][1]['cost_scale']
    return ids, x, same, output


def trial_identity(identity, meta, ids, x, same, y, pr):
    return dict(identity=identity, outer_site=meta['outer_site'], seed=meta['seed'],
        features_sha256=array_hash(x, same), labels_sha256=array_hash(y), ids_sha256=array_hash(ids),
        preprocess_sha256=array_hash(pr['mean'], pr['std'], pr['known'], pr['weights'], pr['constant'], np.array([pr['cost_scale']])),
        training_sites=pr['training_sites'], training_producers=meta['training_producers'])


def completed(path, ti, reg, arm):
    record = json.loads(path.read_text())
    if record['identity'] != ti or record['arm'] != arm or record['fit']['step'] != reg['training']['steps'] or not record['fit']['complete']:
        raise ValueError('Changed or incomplete risk endpoint')
    if file_digest(ROOT/record['checkpoint']) != record['checkpoint_sha256']:
        raise ValueError('Changed risk checkpoint')
    return record


def train(reg, data, views, identity, beat, args):
    root = ROOT/reg['output']
    if args.view and args.view not in views:
        raise ValueError('Unregistered source view')
    for key, meta in views.items():
        if args.view and args.view != key:
            continue
        ids, x, same, arms = training_data(meta, data)
        for arm, (y, pr) in arms.items():
            if args.arm and args.arm != arm:
                continue
            directory = root/'trials'/key/arm
            ti = trial_identity(identity, meta, ids, x, same, y, pr)
            rp = directory/'complete.json'
            if rp.exists():
                completed(rp, ti, reg, arm); beat(state='cached_verified_complete', view=key, arm=arm); continue
            beat(state='training_fixed_risk_head', view=key, arm=arm, supported_rows=int(pr['known'].sum()))
            _, fit = fit_head(x, y, data['sites'][ids], same, pr, seed=meta['seed'], arm=arm,
                settings=reg['training'], identity=ti, directory=directory, resume=args.resume, stop_at=args.stop_at,
                heartbeat=lambda **v:beat(view=key, arm=arm, **v))
            if not fit['complete']:
                beat(state='pilot_complete', view=key, arm=arm, fit=fit); return
            cp = directory/'checkpoint.pt'
            assert_current(identity)
            immutable_json(rp, dict(identity=ti, arm=arm, fit=fit, checkpoint=str(cp.relative_to(ROOT)),
                checkpoint_sha256=file_digest(cp), supported_rows=int(pr['known'].sum()),
                positive_target_rows=int((y[pr['known'], 0] > 0).sum()), risk_calibrated=False))
    beat(state='registered_training_call_complete')


def evaluate(reg, parent, data, views, gain, identity, beat, verify):
    root, public = ROOT/reg['output'], ROOT/reg['reports']
    records = {}
    for key in views:
        for arm in reg['arms']:
            path = root/'trials'/key/arm/'complete.json'
            r = json.loads(path.read_text())
            if r['identity']['identity'] != identity:
                raise ValueError('Mixed experiment endpoint')
            records[key, arm] = completed(path, r['identity'], reg, arm)
    n = len(data['sites']); full = data['valid'].all(1)
    baseline = data['geometry'][:, 332:356].reshape(-1, 12, 2)
    cv, cf = native_errors(baseline, data['target'], data['valid'], data['scale'])
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    quality, pools, bindings, arrays, replay_rows, paired = [], [], [], {}, 0, 0
    for key, meta in views.items():
        train_ids, train_x, train_same, targets = training_data(meta, data)
        site, seed = meta['outer_site'], meta['seed']
        sb = next(r for r in gain['score_archives'] if r['view'] == key)
        with np.load(ROOT/sb['path'], allow_pickle=False) as z:
            ids, gain_scores, asym = z['ids'].copy(), z['mse'].copy(), z['underharm4'].copy()
        with np.load(ROOT/meta['outer_prediction']['prediction']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); pred = z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == site))
        x, same = risk_features(data['geometry'][ids], pred, data['scale'][ids])
        ne, nf = native_errors(pred, data['target'][ids], data['valid'][ids], data['scale'][ids])
        h = np.maximum(ne-cv[ids], 0)
        outputs, sample_draws = {}, None
        for arm in reg['arms']:
            y, pr = targets[arm]
            ti = trial_identity(identity, meta, train_ids, train_x, train_same, y, pr)
            r = completed(root/'trials'/key/arm/'complete.json', ti, reg, arm)
            cp = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
            if cp['identity'] != ti or cp['step'] != reg['training']['steps'] or cp['settings'] != reg['training']:
                raise ValueError('Invalid checkpoint identity')
            for field in ('mean', 'std', 'weights', 'constant', 'known'):
                np.testing.assert_array_equal(cp['preprocess'][field], pr[field])
            if cp['draws'][~pr['known']].sum() != 0 or cp['draws'].sum() != reg['training']['steps']*reg['training']['batch_size']:
                raise ValueError('Wrong training exposure')
            if sample_draws is None:
                sample_draws = cp['draws']
            else:
                np.testing.assert_array_equal(sample_draws, cp['draws']); paired += 1
            model = build_head(reg['feature_dim'], reg['training']['width'], seed)
            model.load_state_dict(cp['model'])
            outputs[arm] = predict_head(model, x, same, pr)
            y_eval = risk_targets(cv[ids], h, full[ids], arm)
            y_zero = risk_targets(cv[ids], h, full[ids], 'zero_reference_harm')
            constant = np.tile(pr['constant'], (len(ids), 1)); constant[same] = 0
            quality.append(dict(view=key, arm=arm,
                own_event=event_quality(outputs[arm][:, 0], y_eval[:, 0], full[ids]),
                own_constant=event_quality(constant[:, 0], y_eval[:, 0], full[ids]),
                zero_event_ranking=event_quality(outputs[arm][:, 0], y_zero[:, 0], full[ids]),
                zero_event_score_is_intended_probability=arm == 'zero_reference_harm',
                own_cost_mse=float(np.mean((outputs[arm][full[ids], 1]-y_eval[full[ids], 1])**2)),
                own_constant_cost_mse=float(np.mean((constant[full[ids], 1]-y_eval[full[ids], 1])**2))))
        path = root/'scores'/(key+'.npz'); receipt = root/'scores'/(key+'.json')
        if verify and not path.exists():
            raise ValueError('Verify cannot create missing risk scores')
        if receipt.exists() and file_digest(path) != json.loads(receipt.read_text())['sha256']:
            raise ValueError('Changed cached risk scores')
        write_arrays(path, dict(ids=ids, **outputs))
        score_record = dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path),
            ids_sha256=array_hash(ids), features_sha256=array_hash(x, same))
        immutable_json(receipt, score_record); bindings.append(score_record)
        replay_rows += len(ids)*len(outputs)
        anchor = fixed_interventions(asym, same)['harm_fraction_0p1']
        xy = data['geometry'][ids, :16].reshape(-1, 8, 2)
        policies, capacity = risk_decisions(gain_scores, outputs, same, np.all(xy[:, -1] == xy[:, -2], axis=1), ids, anchor, reg['risk_score_cut'])
        policies['parent_asym_strict'] = anchor
        policies['parent_mse_strict'] = fixed_interventions(gain_scores, same)['harm_fraction_0p1']
        policies['native_without_guard'] = ~same
        prior_ridge = next(r for r in gain['training'] if r['view'] == key and r['arm'] == 'ridge')
        prior_pr = torch.load(ROOT/prior_ridge['checkpoint'], map_location='cpu', weights_only=False)['preprocess']
        masks['hard'][ids] = cv[ids] >= prior_pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= prior_pr['positive_easy_cut'])
        pools.append(dict(view=key, **capacity, selected_ids_sha256={k:array_hash(ids[v]) for k,v in policies.items()}))
        for name, use in policies.items():
            index = name, seed
            if index not in arrays:
                arrays[index] = dict(ade=np.full(n, np.nan), fde=np.full(n, np.nan), use=np.zeros(n, bool), seen=np.zeros(n, bool))
            d = arrays[index]
            if d['seen'][ids].any():
                raise ValueError('Repeated outer query')
            d['ade'][ids], d['fde'][ids] = np.where(use, ne, cv[ids]), np.where(use, nf, cf[ids])
            d['use'][ids], d['seen'][ids] = use, True
        beat(state='held_source_risk_readout', view=key, common_count=capacity['common_count'])
    def metric(a, b, mask=None):
        if mask is None:
            mask = np.ones(n, bool)
        return paired_scene_metrics(a[mask], b[mask], data['sites'][mask], expected_scenes=parent['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=3000, seed=38113)
    summary = {}
    for policy in sorted({k[0] for k in arrays}):
        ss = {}
        for seed in parent['seeds']:
            d = arrays[policy, seed]
            if not d['seen'].all():
                raise ValueError('Missing outer evaluation query')
            ss[str(seed)] = dict(ADE=metric(d['ade'], cv), FDE=metric(d['fde'], cf),
                subsets={k:metric(d['ade'], cv, v) for k,v in masks.items()},
                intervention_rate=float(d['use'].mean()),
                zero_CV_harmed_rows=int((d['ade'][masks['zero_CV']] > 0).sum()),
                zero_CV_max_absolute_harm=float(d['ade'][masks['zero_CV']].max()),
                unknown_ADE_selected_rows=int((d['use'] & ~np.isfinite(cv)).sum()),
                unknown_complete_risk_selected_rows=int((d['use'] & ~full).sum()))
        mean = np.mean([arrays[policy, s]['ade'] for s in parent['seeds']], axis=0)
        summary[policy] = dict(seeds=ss, ADE=metric(mean, cv), subsets={k:metric(mean, cv, v) for k,v in masks.items()})
    contrasts = {}
    for control in ('all_harm_guard_matched', 'net_only_matched', 'stop_veto_matched'):
        aa = [summary['zero_harm_guard_matched']['ADE']['by_scene'][s]['gain_percent'] for s in parent['sites']]
        bb = [summary[control]['ADE']['by_scene'][s]['gain_percent'] for s in parent['sites']]
        contrasts[control] = paired_scene_contrast(aa, bb)
    result = dict(identity=identity, result_source='fresh_run_risk_head_training_and_source_readout_cached_verified_forecasters_gain_heads',
        training=[dict(view=k, arm=a, fit=r['fit'], checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256']) for (k,a),r in records.items()],
        optimizer_updates=sum(r['fit']['step'] for r in records.values()), summed_fit_seconds=sum(r['fit']['seconds'] for r in records.values()),
        paired_sampler_checks=paired, quality=quality, summary=summary, matched_contrasts=contrasts,
        policy_capacities=pools, score_archives=bindings, indexed_source_rows=n,
        incomplete_labels_not_negative=True, threshold_search=False, independent_confirmation=False,
        risk_calibrated=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity); immutable_json(public/'analysis.json', result)
    if verify:
        immutable_json(public/'verification.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
            risk_score_rows_replayed=replay_rows, checked_heads=len(records), preprocessing_views_recomputed=len(views),
            paired_sampler_checks=paired, all_checks_passed=True, new_training=False))
    beat(state='evaluation_verified' if verify else 'evaluation_complete', heads=len(records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-only', action='store_true'); p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluate', action='store_true'); p.add_argument('--verify', action='store_true')
    p.add_argument('--view'); p.add_argument('--arm', choices=('all_harm', 'zero_reference_harm')); p.add_argument('--stop-at', type=int)
    args = p.parse_args()
    if args.stop_at is not None and (not args.view or not args.arm or args.evaluate or args.verify):
        raise ValueError('Training pilot must identify one fixed view/arm')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, parent, data, views, gain, identity = load()
    root = ROOT/reg['output']; root.mkdir(parents=True, exist_ok=True)
    lock = (root/'execution.lock').open('a'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    def beat(**v):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **v)
        json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    if args.audit_only:
        beat(state='preflight_pass', bindings=len(identity['source_bindings']), views=len(views), rows=len(data['sites'])); return
    immutable_json(root/'identity.json', identity)
    if args.evaluate or args.verify:
        evaluate(reg, parent, data, views, gain, identity, beat, args.verify)
    else:
        train(reg, data, views, identity, beat, args)


if __name__ == '__main__':
    main()
