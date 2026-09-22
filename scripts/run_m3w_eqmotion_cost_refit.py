"""Fixed predictor-specific cost refit with a frozen transfer comparator."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 environment required before Torch import')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts import run_m3w_cost_head_transfer as transfer
from scripts import run_m3w_bounded_cost as bounded
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write
from scripts.run_m3w_native_nested import write_arrays
from src.world_model.m3w_native_nested import require_source_producer
from src.world_model.m3w_eqmotion_cost_data import assemble_view, validate_training_labels
from src.world_model.m3w_bounded_cost_head import ARMS, build, predict
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast

CONFIG = 'configs/m3w_eqmotion_cost_refit_v1.json'
CODE = ('scripts/run_m3w_eqmotion_cost_refit.py', 'src/world_model/m3w_eqmotion_cost_data.py',
        'tests/test_m3w_eqmotion_cost_data.py', 'tests/test_m3w_eqmotion_cost_refit.py',
        'scripts/verify_m3w_eqmotion_cost_refit.py', 'tests/test_m3w_eqmotion_cost_refit_verification.py')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    old_cfg, data, old_views, fit_cfg, heads, predictions, old_id = transfer.load()
    if (cfg['sites'] != old_cfg['sites'] or cfg['seeds'] != old_cfg['seeds']
            or cfg['training'] != fit_cfg['training'] or tuple(cfg['arms']) != ARMS
            or cfg['policies'] != ['net_stop', 'strict_stop', 'matched_count']
            or cfg['matched_count_reference'] != 'frozen_bounded_fraction_strict_stop'
            or cfg['primary_contrast'] != 'refit_minus_frozen_bounded_fraction_strict_stop'
            or cfg['strict_harm_benefit_ratio'] != .1 or not cfg['complete_training_supervision_only']
            or any(cfg[k] for k in ('threshold_search', 'model_selection', 'risk_calibration',
                'independent_confirmation', 'closed_role_readout', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Registered source-only refit and unchanged selection rules required')
    if set(data) & {'target', 'valid', 'future_endpoint'}:
        raise ValueError('No outcome arrays in causal data loader')
    nested = json.loads((ROOT/cfg['nested_analysis']).read_text())
    manifest = json.loads((ROOT/cfg['nested_manifest']).read_text())
    assert manifest['identity'] == nested['identity']
    assert nested['completed_fits'] == 18 and nested['optimizer_updates'] == 72000
    assert file_digest(ROOT/cfg['nested_manifest']) == nested['views_manifest_sha256']
    bindings = dict(old_id['source_bindings'])
    def bind(path, expected=None):
        actual = file_digest(ROOT/path)
        if (expected is not None and actual != expected) or (path in bindings and bindings[path] != actual):
            raise ValueError('Changed dependency: '+path)
        bindings[path] = actual
    for p, sha in nested['identity']['source_bindings'].items(): bind(p, sha)
    for p in (CONFIG, cfg['registration'], cfg['nested_analysis'], cfg['nested_manifest'], *CODE): bind(p)
    for folder, analysis in ((Path(cfg['nested_analysis']).parent, cfg['nested_analysis']),
            (Path(old_cfg['reports']), str(Path(old_cfg['reports'])/'analysis.json'))):
        bind(analysis)
        for name in ('independent_verification.json',
                     'verification_with_replay.json' if folder == Path(cfg['nested_analysis']).parent else 'replay.json'):
            path = str(folder/name); record = json.loads((ROOT/path).read_text())
            assert record['all_checks_passed'] and record['analysis_sha256'] == file_digest(ROOT/analysis)
            bind(path)
    for r in nested['training']: bind(r['checkpoint'], r['checkpoint_sha256'])
    old_report = json.loads((ROOT/old_cfg['reports']/'analysis.json').read_text())
    assert old_report['identity'] == old_id
    frozen_decisions = {r['view']:r for r in old_report['archives']}
    for r in frozen_decisions.values(): bind(r['path'], r['sha256'])
    views = {}
    for view in manifest['views']:
        site, seed = view['outer_site'], view['seed']; key = f'{site}_seed{seed}'
        assert key in old_views and view['outer_producer']['prediction'] == predictions[key]
        groups = []
        for g in view['groups']:
            require_source_producer(g['producer'], outer_site=site, row_site=g['inner_site'],
                roster=cfg['sites'], seed=seed)
            assert g['producer']['family'] == 'eqmotion_fixed_head'
            cache = g['cache']
            for kind in ('prediction', 'supervision'): bind(cache[kind+'_path'], cache[kind+'_sha256'])
            with np.load(ROOT/cache['prediction_path'], allow_pickle=False) as z:
                assert set(z.files) == {'ids', 'prediction'}
                ids, p = z['ids'].copy(), z['prediction'].copy()
            with np.load(ROOT/cache['supervision_path'], allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], ids)
                labels = {k:z[k].copy() for k in ('benefit', 'harm', 'baseline_ade', 'complete_future')}
            groups.append(dict(inner_site=g['inner_site'], ids=ids, prediction=p, labels=labels))
        inputs, targets = assemble_view(data['sites'], site, groups)
        validate_training_labels(targets, bounded.read_arrays(data, inputs['ids'], 'valid').all(1))
        ip = f"{cfg['output']}/training_inputs/{key}.npz"
        tp = f"{cfg['output']}/training_targets/{key}.npz"
        write_arrays(ROOT/ip, inputs); write_arrays(ROOT/tp, targets)
        bind(ip); bind(tp)
        views[key] = dict(outer_site=site, seed=seed, inputs_path=ip, targets_path=tp,
            inputs_sha256=bindings[ip], targets_sha256=bindings[tp],
            training_producers=[g['producer'] for g in view['groups']])
    assert set(views) == set(old_views) == set(frozen_decisions)
    identity = dict(source_bindings=bindings, config=cfg, frozen_transfer_identity=old_id,
        nested_identity=nested['identity'], torch=torch.__version__, numpy=np.__version__,
        architecture=platform.machine(), target_arrays_loaded_for_decisions=False)
    assert_current(identity)
    return cfg, data, views, heads, predictions, frozen_decisions, identity


def evaluate(cfg, data, views, old_heads, predictions, frozen_decisions, identity, beat, verify=False):
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']
    n = len(data['sites']); archives, records = [], {}
    for key in views:
        for arm in ARMS:
            r = json.loads((root/'trials'/key/arm/'complete.json').read_text())
            assert r['identity']['identity'] == identity and r['fit']['complete']
            assert r['fit']['step'] == cfg['training']['steps'] and r['arm'] == arm
            assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
            records[key, arm] = r
    bits, candidates, distances, scores, cuts, shifts = {}, {}, {}, {}, {}, []
    for key, meta in views.items():
        seed = meta['seed']
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            ids, p = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        x, d, same = bounded.features(data['geometry'][ids], p, data['scale'][ids])
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        old_cp = torch.load(ROOT/old_heads[key, 'bounded_fraction']['checkpoint'], map_location='cpu', weights_only=False)
        old_pr = old_cp['preprocess']; cuts[key] = old_pr
        with np.load(ROOT/frozen_decisions[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            np.testing.assert_array_equal(z['distance'], d)
            anchor = z['bounded_fraction_strict_stop'].copy()
            old_scores = {a:z[a].copy() for a in ARMS}
            old_bits = {k:z[k].copy() for k in ('floor', 'uncontrolled', 'past_stop')}
            for arm in ARMS:
                for policy in cfg['policies']:
                    old_bits['frozen_'+arm+'_'+policy] = z[arm+'_'+policy].copy()
        arrays = dict(ids=ids, distance=d, **old_bits)
        choices = dict(old_bits); draws = None
        for arm in ARMS:
            cp = torch.load(ROOT/records[key, arm]['checkpoint'], map_location='cpu', weights_only=False)
            assert cp['identity'] == records[key, arm]['identity'] and cp['arm'] == arm and cp['seed'] == seed
            assert cp['step'] == cfg['training']['steps'] and cp['settings'] == cfg['training']
            pr = cp['preprocess']
            assert set(pr['training_sites']) == set(cfg['sites'])-{meta['outer_site']}
            assert pr['hard_cut'] == old_pr['hard_cut'] and pr['positive_easy_cut'] == old_pr['positive_easy_cut']
            assert cp['draws'][~pr['known']].sum() == 0
            assert cp['draws'].sum() == cfg['training']['steps']*cfg['training']['batch_size']
            if draws is None: draws = cp['draws']
            else: np.testing.assert_array_equal(draws, cp['draws'])
            model = build(x.shape[1], cfg['training']['width'], seed); model.load_state_dict(cp['model'])
            score = predict(model, x, d, pr, arm)
            assert np.isfinite(score).all() and (score >= 0).all() and not score[same].any()
            if arm != 'direct_native': assert np.all(score.sum(1) <= d+2e-6*(1+d))
            scores[key, 'refit_'+arm], scores[key, 'frozen_'+arm] = score, old_scores[arm]
            arrays['refit_'+arm], arrays['frozen_'+arm] = score, old_scores[arm]
            for policy, selected in bounded.selections(score, past, d, anchor, ids).items():
                choices['refit_'+arm+'_'+policy] = selected
                arrays['refit_'+arm+'_'+policy] = selected
            z = np.abs((x.astype(float)-pr['mean'])/pr['std'])
            shifts.append(dict(view=key, arm=arm, refit_feature_abs_z_gt10=float((z > 10).mean()),
                refit_max_abs_z=float(z.max()), frozen_feature_abs_z_gt10=float(
                    (np.abs((x.astype(float)-old_pr['mean'])/old_pr['std']) > 10).mean())))
        path = root/'decisions'/f'{key}.npz'
        if verify and not path.exists(): raise ValueError('Cannot verify missing decisions')
        write_arrays(path, arrays)
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
        if seed not in bits:
            bits[seed] = {name:np.zeros(n, bool) for name in choices}
            candidates[seed] = np.empty((n, 12, 2), p.dtype); distances[seed] = np.empty(n)
        for name, selected in choices.items(): bits[seed][name][ids] = selected
        candidates[seed][ids], distances[seed][ids] = p, d
        beat(state='replayed' if verify else 'decisions_frozen', view=key, rows=len(ids))
    immutable_json(root/'decisions_complete.json', dict(identity=identity, archives=archives,
        future_targets_used_in_decisions=False, score_rows=n*len(cfg['seeds'])*len(ARMS)))
    y, valid = bounded.read_arrays(data, np.arange(n), 'target'), bounded.read_arrays(data, np.arange(n), 'valid')
    b = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = native_errors(b, y, valid, data['scale']); full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    errors = {s:native_errors(p, y, valid, data['scale']) for s,p in candidates.items()}
    bounds = {s:partial_gain_bounds(p, b, y, valid, data['scale']) for s,p in candidates.items()}
    quality = []
    for key, meta in views.items():
        ids = np.flatnonzero(data['sites'] == meta['outer_site']); pr = cuts[key]; seed = meta['seed']
        masks['hard'][ids] = cv[ids] >= pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= pr['positive_easy_cut'])
        delta = cv[ids]-errors[seed][0][ids]; cost = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0)))
        for family in ('frozen', 'refit'):
            for arm in ARMS:
                name = family+'_'+arm; use = full[ids] & bits[seed][name+'_strict_stop'][ids]
                score = scores[key, name]
                quality.append(dict(view=key, head=name, selected_complete=int(use.sum()),
                    complete_cost_MSE=float(((score[full[ids]]-cost[full[ids]])**2).mean()),
                    predicted_harm=None if not use.any() else float(score[use, 1].mean()),
                    realized_harm=None if not use.any() else float(cost[use, 1].mean())))
    def metric(a, r, mask=None):
        if mask is None: mask = np.ones(n, bool)
        return paired_scene_metrics(a[mask], r[mask], data['sites'][mask], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=cfg['bootstrap_resamples'])
    summaries = {}
    for name in next(iter(bits.values())):
        seeds, ades, fdes = {}, [], []
        for seed in cfg['seeds']:
            use = bits[seed][name]
            ade, fde = np.where(use, errors[seed][0], cv), np.where(use, errors[seed][1], cf)
            ades.append(ade); fdes.append(fde)
            seeds[str(seed)] = dict(ADE=metric(ade, cv), FDE=metric(fde, cf), selected=int(use.sum()),
                selected_unknown=int((use & ~valid.any(1)).sum()), selected_incomplete=int((use & ~full).sum()),
                zero_CV_harmed=int((ade[masks['zero_CV']] > 0).sum()),
                full_grid_absolute_gain_bounds={site:[float(np.where(use,bounds[seed][k],0)[data['sites']==site].mean())
                    for k in ('lower', 'upper')] for site in cfg['sites']},
                subsets={g:metric(ade, cv, m) for g,m in masks.items()})
        summaries[name] = dict(ADE=metric(np.mean(ades,0), cv), FDE=metric(np.mean(fdes,0), cf),
            seeds=seeds, subsets={g:metric(np.mean(ades,0), cv,m) for g,m in masks.items()})
    contrasts = {}
    for arm in ARMS:
        for policy in cfg['policies']:
            left, right = [summaries[f'{family}_{arm}_{policy}']['ADE']['by_scene'] for family in ('refit','frozen')]
            contrasts[f'refit_minus_frozen_{arm}_{policy}'] = paired_scene_contrast(
                [left[s]['gain_percent'] for s in cfg['sites']], [right[s]['gain_percent'] for s in cfg['sites']])
    primary = summaries['refit_bounded_fraction_strict_stop']; contrast = contrasts[cfg['primary_contrast']]
    gate = dict(exact_zero=all(v['zero_CV_harmed']==0 for v in primary['seeds'].values()),
        easy=all(-v['subsets']['positive_easy']['equal_scene_gain_percent']<=2 for v in primary['seeds'].values()),
        each_seed_positive_cv=all(v['ADE']['equal_scene_gain_percent']>0 for v in primary['seeds'].values()),
        positive_primary_ci=contrast['ci95_pp'][0]>0)
    result = dict(identity=identity, result_source='fresh_run_36_eqmotion_specific_cost_heads_cached_verified_forecasters',
        archives=archives, summaries=summaries, contrasts=contrasts, conditional_quality=quality, feature_shift=shifts,
        training=[dict(view=key, arm=arm, **{k:r[k] for k in ('fit','rows','supported_rows','checkpoint','checkpoint_sha256')})
                  for (key,arm),r in records.items()], new_fits=len(records),
        score_rows=n*len(cfg['seeds'])*len(ARMS), primary_gates=gate, primary_joint_empirical_pass=all(gate.values()),
        decision_manifest_sha256=file_digest(root/'decisions_complete.json'),
        independent_confirmation=False, risk_calibrated=False, closed_role_readout=False,
        deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity); immutable_json(public/'analysis.json', result)
    if verify:
        immutable_json(public/'replay.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
            cost_heads_replayed=len(records), score_rows=result['score_rows'], all_checks_passed=True))
    beat(state='verified' if verify else 'evaluated', primary_gates=gate)


def validate_args(args):
    readout = args.audit_only or args.evaluate or args.verify
    if (sum((args.audit_only, args.evaluate, args.verify)) > 1
            or (readout and (args.resume or args.view or args.arm or args.stop_at is not None))
            or (args.stop_at is not None and (not args.view or not args.arm or args.stop_at <= 0))):
        raise ValueError('One phase; pilot must name a registered view and arm')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only', 'resume', 'evaluate', 'verify'):
        parser.add_argument('--'+name, action='store_true')
    parser.add_argument('--view'); parser.add_argument('--arm', choices=ARMS); parser.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    validate_args(args)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT/CONFIG).read_text())
    root = ROOT/cfg['output']; root.mkdir(parents=True,exist_ok=True)
    def beat(**v):
        event = dict(pid=os.getpid(),timestamp_unix=time.time(),**v)
        json_write(root/'heartbeat.json',event)
        with (root/'events.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,data,views,heads,predictions,frozen,identity = load()
        if args.view and args.view not in views: raise ValueError('Unregistered view')
        immutable_json(root/'identity.json',identity)
        if args.audit_only: beat(state='preflight_pass',views=len(views),heads=36,bindings=len(identity['source_bindings']))
        elif args.evaluate or args.verify: evaluate(cfg,data,views,heads,predictions,frozen,identity,beat,args.verify)
        else: bounded.train(cfg,data,views,identity,args,beat)


if __name__ == '__main__': main()
