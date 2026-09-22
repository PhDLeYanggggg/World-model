"""Fixed, no-refit transfer of source cost heads to native EqMotion forecasts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
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
from scripts.run_m3w_bounded_cost import load as source_load, features, read_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_cost_head_transfer import choices
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.world_model.m3w_bounded_cost_head import ARMS, build, predict

CONFIG = 'configs/m3w_cost_head_transfer_v1.json'
CODE = ('scripts/run_m3w_cost_head_transfer.py', 'src/evaluation/m3w_cost_head_transfer.py',
        'tests/test_m3w_cost_head_transfer.py')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    source_cfg, data, views, _, parent = source_load()
    if (cfg['sites'] != source_cfg['sites'] or cfg['seeds'] != source_cfg['seeds']
            or tuple(cfg['arms']) != ARMS or cfg['strict_harm_benefit_ratio'] != .1
            or cfg['matched_count_reference'] != 'bounded_fraction_strict_stop'
            or cfg['primary_contrast'] != 'bounded_fraction_strict_stop_minus_direct_native_strict_stop'
            or any(cfg[k] for k in ('new_training', 'threshold_search', 'model_selection',
                'risk_calibration', 'independent_confirmation', 'closed_role_readout',
                'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Frozen source-only transfer design required')
    if set(data) & {'target', 'valid', 'future_endpoint'}:
        raise ValueError('Outcome arrays cannot enter the decision data loader')
    bindings = dict(parent['source_bindings'])
    def bind(path, expected=None):
        actual = file_digest(ROOT/path)
        if (expected is not None and expected != actual) or (path in bindings and bindings[path] != actual):
            raise ValueError('Changed frozen dependency: '+path)
        bindings[path] = actual
    for p, sha in cfg['bindings'].items(): bind(p, sha)
    for p in (CONFIG, cfg['registration'], *CODE): bind(p)
    old_path = ROOT/source_cfg['reports']/'analysis.json'
    old = json.loads(old_path.read_text())
    assert old['identity'] == parent
    ep = ROOT/'outputs/publication_readiness_2026_09/native_eqmotion_v1/analysis.json'
    eq = json.loads(ep.read_text())
    for p, sha in eq['identity']['source_bindings'].items(): bind(p, sha)
    for folder, analysis in ((ep.parent, ep), (old_path.parent, old_path)):
        for name in ('replay.json' if folder == ep.parent else 'verification_with_replay.json',
                     'independent_verification.json'):
            receipt = json.loads((folder/name).read_text())
            assert receipt['all_checks_passed'] and receipt['analysis_sha256'] == file_digest(analysis)
    predictions = {r['view']:r for r in eq['archives']}
    assert set(predictions) == set(views)
    for r in predictions.values(): bind(r['path'], r['sha256'])
    heads = {(r['view'], r['arm']):r for r in old['training']}
    assert set(heads) == {(key, a) for key in views for a in ARMS}
    for r in heads.values(): bind(r['checkpoint'], r['checkpoint_sha256'])
    identity = dict(source_bindings=bindings, config=cfg, source_identity=parent,
        target_prediction_identity=eq['identity'], torch=torch.__version__, numpy=np.__version__,
        architecture=platform.machine(), runtime=dict(torch_threads=4, interop_threads=1, num_workers=0),
        target_arrays_loaded_for_decisions=False)
    assert_current(identity)
    return cfg, data, views, source_cfg, heads, predictions, identity


def run(cfg, data, views, source_cfg, heads, predictions, identity, beat, verify=False):
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']
    n = len(data['sites'])
    selections, candidates, distance, scores, preprocessors = {}, {}, {}, {}, {}
    archives, shifts = [], []
    for key, meta in views.items():
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            assert set(z.files) == {'ids', 'prediction'}
            ids, p = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        x, d, same = features(data['geometry'][ids], p, data['scale'][ids])
        arm_scores = {}
        reference_pr = None
        for arm in ARMS:
            cp = torch.load(ROOT/heads[key, arm]['checkpoint'], map_location='cpu', weights_only=False)
            assert cp['identity']['identity'] == identity['source_identity']
            assert cp['identity']['view'] == key and cp['arm'] == arm and cp['seed'] == meta['seed']
            assert cp['step'] == source_cfg['training']['steps'] and cp['settings'] == source_cfg['training']
            pr = cp['preprocess']
            assert set(pr['training_sites']) == set(cfg['sites'])-{meta['outer_site']}
            if reference_pr is not None:
                for field in ('mean', 'std', 'known', 'weights'):
                    np.testing.assert_array_equal(pr[field], reference_pr[field])
                assert pr['cost_scale'] == reference_pr['cost_scale']
            reference_pr = pr
            model = build(x.shape[1], source_cfg['training']['width'], meta['seed'])
            model.load_state_dict(cp['model'])
            score = predict(model, x, d, pr, arm)
            assert np.isfinite(score).all() and np.all(score >= 0) and not score[same].any()
            if arm != 'direct_native': assert np.all(score.sum(1) <= d+2e-6*(1+d))
            arm_scores[arm] = score
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        bits = choices(arm_scores, past, d, ids)
        arrays = dict(ids=ids, distance=d, **arm_scores, **bits)
        path = root/'decisions'/f'{key}.npz'
        if verify and not path.exists(): raise ValueError('Missing decisions cannot be verified')
        write_arrays(path, arrays)
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
        z = np.abs((x.astype(float)-reference_pr['mean'])/reference_pr['std'])
        shifts.append(dict(view=key, feature_abs_z_gt5_fraction=float((z > 5).mean()),
            feature_abs_z_gt10_fraction=float((z > 10).mean()), max_abs_z=float(z.max()),
            normalization_refitted=False))
        seed = meta['seed']
        if seed not in selections:
            selections[seed] = {name:np.zeros(n, bool) for name in bits}
            candidates[seed] = np.empty((n, 12, 2), p.dtype)
            distance[seed] = np.empty(n)
        for name, selected in bits.items(): selections[seed][name][ids] = selected
        candidates[seed][ids], distance[seed][ids] = p, d
        scores[key], preprocessors[key] = arm_scores, reference_pr
        beat(state='scores_replayed' if verify else 'scores_frozen', view=key, rows=len(ids))
    immutable_json(root/'decisions_complete.json', dict(identity=identity, archives=archives,
        future_targets_used_in_decisions=False, new_training=False, score_rows=n*len(cfg['seeds'])*len(ARMS)))
    # The complete decision manifest precedes loading any evaluation target array.
    target, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    baseline = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = native_errors(baseline, target, valid, data['scale'])
    full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0),
        hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    errors, bounds = {}, {}
    for seed in cfg['seeds']:
        p = candidates[seed]
        errors[seed] = native_errors(p, target, valid, data['scale'])
        bounds[seed] = partial_gain_bounds(p, baseline, target, valid, data['scale'])
    quality = []
    for key, meta in views.items():
        ids = np.flatnonzero(data['sites'] == meta['outer_site'])
        pr, seed = preprocessors[key], meta['seed']
        masks['hard'][ids] = cv[ids] >= pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= pr['positive_easy_cut'])
        delta = cv[ids]-errors[seed][0][ids]
        costs = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0)))
        for arm in ARMS:
            s = scores[key][arm]
            use = full[ids] & selections[seed][arm+'_strict_stop'][ids]
            quality.append(dict(view=key, arm=arm, complete_rows=int(full[ids].sum()),
                complete_cost_MSE=float(((s[full[ids]]-costs[full[ids]])**2).mean()),
                selected_complete=int(use.sum()), predicted_harm=None if not use.any() else float(s[use, 1].mean()),
                realized_harm=None if not use.any() else float(costs[use, 1].mean())))
    def metric(a, b, mask=None):
        if mask is None: mask = np.ones(n, bool)
        return paired_scene_metrics(a[mask], b[mask], data['sites'][mask], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=cfg['bootstrap_resamples'])
    summaries = {}
    for name in next(iter(selections.values())):
        seeds, ades, fdes = {}, [], []
        for seed in cfg['seeds']:
            use = selections[seed][name]
            ade, fde = np.where(use, errors[seed][0], cv), np.where(use, errors[seed][1], cf)
            ades.append(ade); fdes.append(fde)
            seeds[str(seed)] = dict(ADE=metric(ade, cv), FDE=metric(fde, cf), selected=int(use.sum()),
                effective_switches=int((use & (distance[seed] > 0)).sum()),
                selected_unknown=int((use & ~valid.any(1)).sum()), selected_incomplete=int((use & ~full).sum()),
                zero_CV_harmed=int((ade[masks['zero_CV']] > 0).sum()),
                full_grid_absolute_gain_bounds={site:[float(np.where(use, bounds[seed][k], 0)[data['sites'] == site].mean())
                    for k in ('lower', 'upper')] for site in cfg['sites']},
                subsets={g:metric(ade, cv, m) for g,m in masks.items()})
        summaries[name] = dict(ADE=metric(np.mean(ades, 0), cv), FDE=metric(np.mean(fdes, 0), cf), seeds=seeds,
            subsets={g:metric(np.mean(ades, 0), cv, m) for g,m in masks.items()})
    contrasts = {}
    for policy in ('strict_stop', 'matched_count'):
        for right in ('direct_native', 'bounded_native'):
            left = summaries['bounded_fraction_'+policy]['ADE']['by_scene']
            ref = summaries[right+'_'+policy]['ADE']['by_scene']
            contrasts['bounded_fraction_minus_'+right+'_'+policy] = paired_scene_contrast(
                [left[s]['gain_percent'] for s in cfg['sites']], [ref[s]['gain_percent'] for s in cfg['sites']])
    primary = summaries['bounded_fraction_strict_stop']
    zero_pass = all(v['zero_CV_harmed'] == 0 for v in primary['seeds'].values())
    easy_pass = all(-v['subsets']['positive_easy']['equal_scene_gain_percent'] <= 2 for v in primary['seeds'].values())
    contrast = contrasts['bounded_fraction_minus_direct_native_strict_stop']
    positive_cv = all(v['ADE']['equal_scene_gain_percent'] > 0 for v in primary['seeds'].values())
    positive_ci = contrast['ci95_pp'][0] > 0
    result = dict(identity=identity, result_source='fresh_run_frozen_cost_head_transfer_cached_verified_models',
        archives=archives, summaries=summaries, contrasts=contrasts, conditional_quality=quality, feature_shift=shifts,
        reused_cost_heads=len(heads), new_fits=0, score_rows=n*len(cfg['seeds'])*len(ARMS),
        decision_manifest_sha256=file_digest(root/'decisions_complete.json'),
        primary_exact_zero_pass=zero_pass, primary_easy_pass=easy_pass,
        primary_positive_contrast=contrast['mean_gain_difference_pp'] > 0,
        primary_each_seed_positive_vs_CV=positive_cv, primary_contrast_ci_low_positive=positive_ci,
        primary_joint_empirical_pass=zero_pass and easy_pass and positive_cv and positive_ci,
        independent_confirmation=False, risk_calibrated=False, closed_role_readout=False,
        deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity)
    immutable_json(public/'analysis.json', result)
    if verify:
        immutable_json(public/'replay.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
            cost_heads_replayed=len(heads), score_rows=result['score_rows'], all_checks_passed=True))
    beat(state='verified' if verify else 'evaluated', analysis_sha256=file_digest(public/'analysis.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.audit_only and args.verify: raise ValueError('One phase per call')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, source_cfg, heads, predictions, identity = load()
    root = ROOT/cfg['output']; root.mkdir(parents=True, exist_ok=True)
    def beat(**v):
        row = dict(pid=os.getpid(), timestamp_unix=time.time(), **v)
        json_write(root/'heartbeat.json', row)
        print(json.dumps(row), flush=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable_json(root/'identity.json', identity)
        if args.audit_only:
            beat(state='preflight_pass', bindings=len(identity['source_bindings']), cost_heads=len(heads), views=len(views))
        else:
            run(cfg, data, views, source_cfg, heads, predictions, identity, beat, verify=args.verify)


if __name__ == '__main__':
    main()
