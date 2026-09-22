"""One fixed loss-weight change with unchanged forecasts and decision rules."""
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
    raise RuntimeError('Native arm64 required before importing Torch')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_eqmotion_cost_refit as parent
from scripts import run_m3w_bounded_cost as bounded
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write, array_hash
from scripts.run_m3w_native_nested import write_arrays
from src.world_model.m3w_tempered_cost_head import fit
from src.world_model.m3w_bounded_cost_head import build, predict
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds

CONFIG = 'configs/m3w_tempered_cost_v1.json'
CODE = ('scripts/run_m3w_tempered_cost.py', 'scripts/verify_m3w_tempered_cost.py',
        'src/world_model/m3w_tempered_cost_head.py', 'tests/test_m3w_tempered_cost_head.py',
        'tests/test_m3w_tempered_cost_protocol.py')
POLICIES = ('net_stop', 'strict_stop', 'matched_count')


def validate_config(cfg, pcfg):
    if (cfg['loss_exponent'] != 1 or cfg['training'] != pcfg['training']
            or cfg['sites'] != pcfg['sites'] or cfg['seeds'] != pcfg['seeds']
            or cfg['primary_reference'] != 'refit_bounded_fraction_strict_stop'
            or cfg['matched_count_reference'] != 'frozen_bounded_fraction_strict_stop'
            or cfg['bootstrap_resamples'] != 3000 or any(cfg[k] for k in (
                'threshold_search', 'model_selection', 'risk_calibration', 'closed_role_readout',
                'independent_confirmation', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Only the fixed intermediate weighting and unchanged source protocol are allowed')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    pcfg, data, views, _, predictions, _, pid = parent.load()
    validate_config(cfg, pcfg)
    prior = json.loads((ROOT/cfg['parent_analysis']).read_text())
    assert prior['identity'] == pid and prior['new_fits'] == 36
    bindings = dict(pid['source_bindings'])
    def bind(path, expected=None):
        actual = file_digest(ROOT/path)
        if (expected is not None and expected != actual) or (path in bindings and bindings[path] != actual):
            raise ValueError('Changed dependency: '+path)
        bindings[path] = actual
    for path in (CONFIG, cfg['registration'], cfg['parent_analysis'], cfg['fit_diagnostic'], *CODE): bind(path)
    for name in ('replay.json', 'independent_verification.json'):
        path = str(Path(cfg['parent_analysis']).parent/name)
        result = json.loads((ROOT/path).read_text())
        assert result['all_checks_passed'] and result['analysis_sha256'] == bindings[cfg['parent_analysis']]
        bind(path)
    diagnostic = json.loads((ROOT/cfg['fit_diagnostic']).read_text())
    for path, sha in diagnostic['identity']['source_bindings'].items(): bind(path, sha)
    for r in prior['training']: bind(r['checkpoint'], r['checkpoint_sha256'])
    for r in prior['archives']: bind(r['path'], r['sha256'])
    identity = dict(source_bindings=bindings, config=cfg, parent_identity=pid,
        torch=torch.__version__, numpy=np.__version__, architecture=platform.machine(),
        outcome_arrays_in_inference=False, all_four_sources_design_exposed=True)
    assert_current(identity)
    return cfg, data, views, predictions, prior, identity


def train(cfg, data, views, prior, identity, args, beat):
    references = {r['view']:r for r in prior['training'] if r['arm'] == 'bounded_fraction'}
    for key, meta in views.items():
        if args.view and key != args.view: continue
        ids, x, y, d, pr = bounded.training_data(meta, data)
        ti = dict(identity=identity, view=key, inputs_sha256=array_hash(ids, x, d), labels_sha256=array_hash(y),
            complete_mask_sha256=array_hash(pr['known']), training_sites=pr['training_sites'],
            producers=meta['training_producers'])
        folder = ROOT/cfg['output']/'trials'/key; receipt = folder/'complete.json'
        if receipt.exists():
            r = json.loads(receipt.read_text())
            assert r['identity'] == ti and r['fit']['complete']
            assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
            beat(state='cached_verified_complete', view=key); continue
        _, result = fit(x, y, d, data['sites'][ids], pr, seed=meta['seed'], settings=cfg['training'],
            identity=ti, directory=folder, resume=args.resume, stop_at=args.stop_at,
            heartbeat=lambda **v:beat(view=key, **v))
        if not result['complete']:
            beat(state='pilot_complete_not_full_fit', view=key); return
        cp_path = folder/'checkpoint.pt'
        cp = torch.load(cp_path, map_location='cpu', weights_only=False)
        old = torch.load(ROOT/references[key]['checkpoint'], map_location='cpu', weights_only=False)
        np.testing.assert_array_equal(cp['draws'], old['draws'])
        for field in ('mean', 'std', 'known', 'weights', 'constant'):
            np.testing.assert_array_equal(pr[field], old['preprocess'][field])
        assert pr['cost_scale'] == old['preprocess']['cost_scale']
        assert_current(identity)
        immutable_json(receipt, dict(identity=ti, fit=result, rows=len(ids), supported_rows=int(pr['known'].sum()),
            checkpoint=str(cp_path.relative_to(ROOT)), checkpoint_sha256=file_digest(cp_path),
            paired_draws_verified=True, independent_confirmation=False))
    beat(state='training_call_complete')


def causal_scores(data, ids, candidate, cp, cfg):
    if set(data) & {'target', 'valid', 'future_endpoint', 'complete_future'}:
        raise ValueError('Inference must not receive future outcomes or label support')
    x, d, same = bounded.features(data['geometry'][ids], candidate, data['scale'][ids])
    model = build(x.shape[1], cfg['training']['width'], cp['seed']); model.load_state_dict(cp['model'])
    score = predict(model, x, d, cp['preprocess'], 'bounded_native')
    assert np.isfinite(score).all() and (score >= 0).all() and not score[same].any()
    assert np.all(score.sum(1) <= d+2e-6*(1+d))
    return score, d


def empirical_gate(summary, contrast):
    return dict(exact_zero=all(v['zero_CV_harmed'] == 0 for v in summary['seeds'].values()),
        easy=all(-v['subsets']['positive_easy']['equal_scene_gain_percent'] <= 2 for v in summary['seeds'].values()),
        each_seed_positive_cv=all(v['ADE']['equal_scene_gain_percent'] > 0 for v in summary['seeds'].values()),
        positive_primary_ci=contrast['ci95_pp'][0] > 0)


def evaluate(cfg, data, views, predictions, prior, identity, beat, verify=False):
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']; n = len(data['sites'])
    records, archives, scores, cuts = {}, [], {}, {}
    chosen = {s:{k:np.zeros(n, bool) for k in POLICIES} for s in cfg['seeds']}
    candidates = {s:np.empty((n, 12, 2), np.float32) for s in cfg['seeds']}
    parent_archives = {r['view']:r for r in prior['archives']}
    references = {r['view']:r for r in prior['training'] if r['arm'] == 'bounded_fraction'}
    for key, meta in views.items():
        r = json.loads((root/'trials'/key/'complete.json').read_text()); records[key] = r
        assert r['identity']['identity'] == identity and r['fit']['complete'] and r['paired_draws_verified']
        assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
        cp = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
        assert cp['identity'] == r['identity'] and cp['settings'] == cfg['training']
        assert cp['step'] == cfg['training']['steps'] and cp['seed'] == meta['seed']
        assert cp['loss_exponent'] == 1 and cp['forward_arm'] == 'bounded_native'
        pr = cp['preprocess']; cuts[key] = pr
        old = torch.load(ROOT/references[key]['checkpoint'], map_location='cpu', weights_only=False)
        np.testing.assert_array_equal(cp['draws'], old['draws'])
        assert cp['draws'][~pr['known']].sum() == 0
        assert pr['hard_cut'] == old['preprocess']['hard_cut']
        assert pr['positive_easy_cut'] == old['preprocess']['positive_easy_cut']
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            ids, p = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        score, d = causal_scores(data, ids, p, cp, cfg)
        with np.load(ROOT/parent_archives[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); np.testing.assert_array_equal(z['distance'], d)
            anchor = z[cfg['matched_count_reference']].copy()
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        choices = bounded.selections(score, past, d, anchor, ids)
        path = root/'decisions'/f'{key}.npz'
        if verify and not path.exists(): raise ValueError('Cannot verify missing decisions')
        write_arrays(path, dict(ids=ids, score=score, distance=d, anchor=anchor, **choices))
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
        for name, bits in choices.items(): chosen[meta['seed']][name][ids] = bits
        candidates[meta['seed']][ids] = p; scores[key] = score
        beat(state='replayed' if verify else 'decisions_frozen', view=key, rows=len(ids))
    immutable_json(root/'decisions_complete.json', dict(identity=identity, archives=archives,
        future_targets_used_in_decisions=False, score_rows=n*len(cfg['seeds'])))
    # No held-source target is loaded until every fixed choice is archived.
    y, valid = bounded.read_arrays(data, np.arange(n), 'target'), bounded.read_arrays(data, np.arange(n), 'valid')
    b = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = native_errors(b, y, valid, data['scale']); full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    errors = {s:native_errors(p, y, valid, data['scale']) for s,p in candidates.items()}
    bounds = {s:partial_gain_bounds(p, b, y, valid, data['scale']) for s,p in candidates.items()}
    quality = []
    for key, meta in views.items():
        ids = np.flatnonzero(data['sites'] == meta['outer_site']); pr = cuts[key]
        masks['hard'][ids] = cv[ids] >= pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= pr['positive_easy_cut'])
        delta = cv[ids]-errors[meta['seed']][0][ids]
        target = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0)))
        use = full[ids] & chosen[meta['seed']]['strict_stop'][ids]; score = scores[key]
        quality.append(dict(view=key, selected_complete=int(use.sum()),
            complete_cost_MSE=float(((score[full[ids]]-target[full[ids]])**2).mean()),
            predicted_benefit=None if not use.any() else float(score[use, 0].mean()),
            realized_benefit=None if not use.any() else float(target[use, 0].mean()),
            predicted_harm=None if not use.any() else float(score[use, 1].mean()),
            realized_harm=None if not use.any() else float(target[use, 1].mean())))
    def metric(a, r, mask=None):
        if mask is None: mask = np.ones(n, bool)
        return paired_scene_metrics(a[mask], r[mask], data['sites'][mask], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=cfg['bootstrap_resamples'])
    summaries = {}
    for name in POLICIES:
        seeds, ades, fdes = {}, [], []
        for seed in cfg['seeds']:
            use = chosen[seed][name]
            ade, fde = np.where(use, errors[seed][0], cv), np.where(use, errors[seed][1], cf)
            ades.append(ade); fdes.append(fde)
            seeds[str(seed)] = dict(ADE=metric(ade, cv), FDE=metric(fde, cf), selected=int(use.sum()),
                selected_unknown=int((use & ~valid.any(1)).sum()), selected_incomplete=int((use & ~full).sum()),
                zero_CV_harmed=int((ade[masks['zero_CV']] > 0).sum()),
                full_grid_absolute_gain_bounds={site:[float(np.where(use, bounds[seed][k], 0)[data['sites']==site].mean())
                    for k in ('lower', 'upper')] for site in cfg['sites']},
                subsets={g:metric(ade, cv, m) for g,m in masks.items()})
        summaries[name] = dict(ADE=metric(np.mean(ades, 0), cv), FDE=metric(np.mean(fdes, 0), cf),
            seeds=seeds, subsets={g:metric(np.mean(ades, 0), cv, m) for g,m in masks.items()})
    contrasts = {}
    for policy in POLICIES:
        for arm in ('direct_native', 'bounded_native', 'bounded_fraction'):
            name = 'refit_'+arm+'_'+policy
            a, r = summaries[policy]['ADE']['by_scene'], prior['summaries'][name]['ADE']['by_scene']
            contrasts[policy+'_minus_'+name] = paired_scene_contrast(
                [a[s]['gain_percent'] for s in cfg['sites']], [r[s]['gain_percent'] for s in cfg['sites']],
                resamples=cfg['bootstrap_resamples'])
    primary_key = 'strict_stop_minus_'+cfg['primary_reference']
    gate = empirical_gate(summaries['strict_stop'], contrasts[primary_key])
    result = dict(identity=identity, result_source='fresh_run_12_heads_fixed_readout',
        references_source='cached_verified_parent_models_choices_and_reductions', archives=archives,
        summaries=summaries, contrasts=contrasts, conditional_quality=quality,
        training=[dict(view=k, **{f:r[f] for f in ('fit','rows','supported_rows','checkpoint','checkpoint_sha256')})
            for k,r in records.items()], new_fits=len(records), score_rows=n*len(cfg['seeds']),
        primary_contrast=primary_key, primary_gates=gate, primary_joint_empirical_pass=all(gate.values()),
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
            or (readout and (args.resume or args.view or args.stop_at is not None))
            or (args.stop_at is not None and (not args.view or args.stop_at <= 0))):
        raise ValueError('One phase; pilot must name a registered view')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only', 'resume', 'evaluate', 'verify'): parser.add_argument('--'+name, action='store_true')
    parser.add_argument('--view'); parser.add_argument('--stop-at', type=int)
    args = parser.parse_args(); validate_args(args)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT/CONFIG).read_text()); root = ROOT/cfg['output']; root.mkdir(parents=True, exist_ok=True)
    def beat(**v):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **v)
        json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg, data, views, predictions, prior, identity = load()
        if args.view and args.view not in views: raise ValueError('Unregistered view')
        immutable_json(root/'identity.json', identity)
        if args.audit_only: beat(state='preflight_pass', views=len(views), heads=12, bindings=len(identity['source_bindings']))
        elif args.evaluate or args.verify: evaluate(cfg, data, views, predictions, prior, identity, beat, args.verify)
        else: train(cfg, data, views, prior, identity, args, beat)


if __name__ == '__main__': main()
