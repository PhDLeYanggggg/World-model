"""Separate choice and metric arithmetic for the fixed intermediate-loss trial."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_tempered_cost import load, POLICIES
import numpy as np
import torch
from scripts.run_m3w_bounded_cost import read_arrays, training_data
from scripts.run_m3w_native_forecast import assert_current, immutable_json, array_hash
from scripts.verify_m3w_native_joint_controls import distances, check_reduction
from src.evaluation.m3w_experiment_contract import file_digest


def expected_choices(score, allowed, ids, anchor):
    if (score.shape != (len(ids), 2) or allowed.shape != ids.shape or anchor.shape != ids.shape
            or allowed.dtype.kind != 'b' or anchor.dtype.kind != 'b' or not np.isfinite(score).all()
            or (score < 0).any() or len(np.unique(ids)) != len(ids) or np.any(anchor & ~allowed)):
        raise ValueError('Finite paired scores, unique IDs and supported frozen anchor required')
    net = allowed & (score[:, 0] > score[:, 1])
    strict = net & (score[:, 1] <= .1*score[:, 0])
    order = sorted(np.flatnonzero(allowed), key=lambda i:(float(score[i, 1]-score[i, 0]), int(ids[i])))
    matched = np.zeros(len(ids), bool); matched[order[:int(anchor.sum())]] = True
    return dict(net_stop=net, strict_stop=strict, matched_count=matched)


def check_contrast(a, b, roster, result):
    diff = np.array([a[s]['gain_percent']-b[s]['gain_percent'] for s in roster])
    np.testing.assert_allclose(diff, result['scene_differences_pp'], atol=1e-10)
    np.testing.assert_allclose(diff.mean(), result['mean_gain_difference_pp'], atol=1e-10)
    rng = np.random.default_rng(result['seed'])
    indices = rng.integers(len(roster), size=(result['resamples'], len(roster)))
    np.testing.assert_allclose(np.quantile(diff[indices].mean(1), [.025, .975]), result['ci95_pp'], atol=1e-10)


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, predictions, prior, identity = load()
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']
    report = json.loads((public/'analysis.json').read_text()); replay = json.loads((public/'replay.json').read_text())
    assert report['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(public/'analysis.json')
    assert report['decision_manifest_sha256'] == file_digest(root/'decisions_complete.json')
    n = len(data['sites']); cuts, scores = {}, {}
    chosen = {s:{p:np.zeros(n, bool) for p in POLICIES} for s in cfg['seeds']}
    pred = {s:np.empty((n, 12, 2), np.float32) for s in cfg['seeds']}
    refs = {r['view']:r for r in prior['training'] if r['arm'] == 'bounded_fraction'}
    old_archives = {r['view']:r for r in prior['archives']}
    fits, policies, supervision_rows = 0, 0, 0
    for key, meta in views.items():
        train_ids, x, labels, d, pr = training_data(meta, data)
        np.testing.assert_array_equal(train_ids, np.flatnonzero(data['sites'] != meta['outer_site']))
        with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
            assert set(z.files) == {'ids', 'prediction'}
            np.testing.assert_array_equal(z['ids'], train_ids); train_p = z['prediction'].copy()
        ty, tv = read_arrays(data, train_ids, 'target'), read_arrays(data, train_ids, 'valid')
        tb = data['geometry'][train_ids, 332:356].reshape(-1, 12, 2)
        bc, _ = distances(tb, ty, tv, data['scale'][train_ids])
        ec, _ = distances(train_p, ty, tv, data['scale'][train_ids])
        expected = np.column_stack((np.maximum(bc-ec, 0), np.maximum(ec-bc, 0)))
        expected[~tv.all(1)] = np.nan
        np.testing.assert_allclose(expected, labels, rtol=1e-12, atol=1e-9, equal_nan=True)
        np.testing.assert_array_equal(pr['known'], tv.all(1)); supervision_rows += len(train_ids)
        r = next(r for r in report['training'] if r['view'] == key)
        assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
        cp = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
        old = torch.load(ROOT/refs[key]['checkpoint'], map_location='cpu', weights_only=False)
        assert cp['identity']['identity'] == identity and cp['identity']['inputs_sha256'] == array_hash(train_ids, x, d)
        assert cp['identity']['labels_sha256'] == array_hash(labels)
        assert cp['identity']['complete_mask_sha256'] == array_hash(pr['known'])
        assert cp['step'] == 3000 and cp['settings'] == cfg['training'] and cp['seed'] == meta['seed']
        assert cp['loss_exponent'] == 1 and cp['forward_arm'] == 'bounded_native'
        for field in ('mean', 'std', 'known', 'weights', 'constant'):
            np.testing.assert_array_equal(cp['preprocess'][field], pr[field])
            np.testing.assert_array_equal(cp['preprocess'][field], old['preprocess'][field])
        assert pr['cost_scale'] == cp['preprocess']['cost_scale'] == old['preprocess']['cost_scale']
        assert pr['hard_cut'] == old['preprocess']['hard_cut']
        assert pr['positive_easy_cut'] == old['preprocess']['positive_easy_cut']
        np.testing.assert_array_equal(cp['draws'], old['draws'])
        assert cp['draws'][~pr['known']].sum() == 0 and cp['draws'].sum() == 3000*256
        fits += 1; cuts[key] = pr
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            ids, p = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        diff = p.astype(float)-b.astype(float)
        distance = np.sqrt(diff[..., 0]**2+diff[..., 1]**2).mean(1)*data['scale'][ids]
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        allowed = (distance > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
        with np.load(ROOT/old_archives[key]['path'], allow_pickle=False) as z:
            old_score = z['frozen_bounded_fraction']
            anchor = allowed & (old_score[:, 0] > old_score[:, 1]) & (old_score[:, 1] <= .1*old_score[:, 0])
            np.testing.assert_array_equal(anchor, z[cfg['matched_count_reference']])
        r = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT/r['path']) == r['sha256']
        with np.load(ROOT/r['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); np.testing.assert_array_equal(z['anchor'], anchor)
            np.testing.assert_allclose(z['distance'], distance, rtol=1e-12, atol=1e-10)
            score = z['score'].copy(); scores[key] = score
            for name, bits in expected_choices(score, allowed, ids, anchor).items():
                np.testing.assert_array_equal(z[name], bits); chosen[meta['seed']][name][ids] = bits
                policies += 1
        pred[meta['seed']][ids] = p
    y, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    b = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = distances(b, y, valid, data['scale']); full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    errors, bounds = {}, {}
    for seed, p in pred.items():
        errors[seed] = distances(p, y, valid, data['scale'])
        safe = np.where(valid[..., None], y, 0).astype(float)
        dd = np.sqrt(np.sum((p.astype(float)-b.astype(float))**2, axis=-1))*data['scale'][:, None]
        pe = np.sqrt(np.sum((p.astype(float)-safe)**2, axis=-1))*data['scale'][:, None]
        be = np.sqrt(np.sum((b.astype(float)-safe)**2, axis=-1))*data['scale'][:, None]
        observed, unknown = np.where(valid, be-pe, 0).sum(1)/12, np.where(valid, 0, dd).sum(1)/12
        bounds[seed] = observed-unknown, observed+unknown
    for key, meta in views.items():
        ids = np.flatnonzero(data['sites'] == meta['outer_site']); pr = cuts[key]
        masks['hard'][ids] = cv[ids] >= pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= pr['positive_easy_cut'])
        delta = cv[ids]-errors[meta['seed']][0][ids]
        cost = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0)))
        use = full[ids] & chosen[meta['seed']]['strict_stop'][ids]; score = scores[key]
        r = next(r for r in report['conditional_quality'] if r['view'] == key)
        assert r['selected_complete'] == int(use.sum())
        np.testing.assert_allclose(((score[full[ids]]-cost[full[ids]])**2).mean(), r['complete_cost_MSE'])
        for i, label in enumerate(('benefit', 'harm')):
            if use.any():
                np.testing.assert_allclose([score[use, i].mean(), cost[use, i].mean()],
                    [r['predicted_'+label], r['realized_'+label]], rtol=1e-10, atol=1e-10)
            else: assert r['predicted_'+label] is None and r['realized_'+label] is None
    reductions = 0
    for name in POLICIES:
        ades, fdes = [], []
        for seed in cfg['seeds']:
            use = chosen[seed][name]
            ade, fde = np.where(use, errors[seed][0], cv), np.where(use, errors[seed][1], cf)
            ades.append(ade); fdes.append(fde); r = report['summaries'][name]['seeds'][str(seed)]
            assert int(use.sum()) == r['selected'] and int((use & ~valid.any(1)).sum()) == r['selected_unknown']
            assert int((use & ~full).sum()) == r['selected_incomplete']
            assert int((ade[masks['zero_CV']] > 0).sum()) == r['zero_CV_harmed']
            reductions += check_reduction(ade, cv, data['sites'], cfg['sites'], r['ADE'])
            reductions += check_reduction(fde, cf, data['sites'], cfg['sites'], r['FDE'])
            for group, mask in masks.items():
                reductions += check_reduction(ade[mask], cv[mask], data['sites'][mask], cfg['sites'], r['subsets'][group])
            for site in cfg['sites']:
                np.testing.assert_allclose([np.where(use, v, 0)[data['sites']==site].mean() for v in bounds[seed]],
                    r['full_grid_absolute_gain_bounds'][site], rtol=1e-10, atol=1e-10)
        r = report['summaries'][name]
        reductions += check_reduction(np.mean(ades, 0), cv, data['sites'], cfg['sites'], r['ADE'])
        reductions += check_reduction(np.mean(fdes, 0), cf, data['sites'], cfg['sites'], r['FDE'])
        for group, mask in masks.items():
            reductions += check_reduction(np.mean(ades, 0)[mask], cv[mask], data['sites'][mask], cfg['sites'], r['subsets'][group])
        for arm in ('direct_native', 'bounded_native', 'bounded_fraction'):
            key = 'refit_'+arm+'_'+name
            check_contrast(r['ADE']['by_scene'], prior['summaries'][key]['ADE']['by_scene'], cfg['sites'],
                report['contrasts'][name+'_minus_'+key])
    p = report['summaries']['strict_stop']
    gate = dict(exact_zero=all(r['zero_CV_harmed'] == 0 for r in p['seeds'].values()),
        easy=all(-r['subsets']['positive_easy']['equal_scene_gain_percent'] <= 2 for r in p['seeds'].values()),
        each_seed_positive_cv=all(r['ADE']['equal_scene_gain_percent'] > 0 for r in p['seeds'].values()),
        positive_primary_ci=report['contrasts'][report['primary_contrast']]['ci95_pp'][0] > 0)
    assert gate == report['primary_gates'] and all(gate.values()) == report['primary_joint_empirical_pass']
    result = dict(analysis_sha256=file_digest(public/'analysis.json'), verifier_sha256=file_digest(Path(__file__)),
        all_checks_passed=True, cost_fits_checked=fits, supervision_rows_checked=supervision_rows,
        policies_checked=policies, scene_reductions=reductions, scores_replayed_by_runner=n*len(cfg['seeds']),
        independent_implementation_same_agent=True, independent_research_confirmation=False,
        preprocessing_recomputed_with_shared_helper=True, new_training=False, selection_changes=False, deployment=False)
    assert_current(identity); immutable_json(public/'independent_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
