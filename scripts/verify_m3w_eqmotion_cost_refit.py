"""Separate arithmetic for predictor-specific fitting and fixed intervention."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_m3w_eqmotion_cost_refit import load
from scripts.run_m3w_bounded_cost import read_arrays, training_data
from scripts.run_m3w_native_forecast import assert_current, immutable_json, array_hash
from scripts.verify_m3w_native_joint_controls import distances, check_reduction
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_bounded_cost_head import ARMS


def expected_choices(scores, allowed, ids):
    """Reconstruct all six families using one frozen, not refitted, count."""
    expected = dict(floor=np.zeros(len(ids), bool), uncontrolled=np.ones(len(ids), bool), past_stop=allowed)
    for name, score in scores.items():
        if score.shape != (len(ids), 2) or not np.isfinite(score).all() or (score < 0).any():
            raise ValueError('Finite nonnegative paired scores required')
        expected[name+'_net_stop'] = allowed & (score[:, 0] > score[:, 1])
        expected[name+'_strict_stop'] = expected[name+'_net_stop'] & (score[:, 1] <= .1*score[:, 0])
    count = int(expected['frozen_bounded_fraction_strict_stop'].sum())
    for name, score in scores.items():
        order = sorted(np.flatnonzero(allowed), key=lambda i:(float(score[i, 1]-score[i, 0]), int(ids[i])))
        bits = np.zeros(len(ids), bool); bits[order[:count]] = True
        expected[name+'_matched_count'] = bits
    return expected


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, heads, predictions, frozen, identity = load()
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']
    report = json.loads((public/'analysis.json').read_text())
    replay = json.loads((public/'replay.json').read_text())
    assert report['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(public/'analysis.json')
    assert file_digest(root/'decisions_complete.json') == report['decision_manifest_sha256']
    n = len(data['sites']); names = list(report['summaries'])
    chosen = {s:{k:np.zeros(n, bool) for k in names} for s in cfg['seeds']}
    pred = {s:np.empty((n, 12, 2), np.float32) for s in cfg['seeds']}
    cuts, scores, fit_checks, choice_checks = {}, {}, 0, 0
    for key, meta in views.items():
        train_ids, x, y, d, pr = training_data(meta, data)
        np.testing.assert_array_equal(train_ids, np.flatnonzero(data['sites'] != meta['outer_site']))
        with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
            assert set(z.files) == {'ids', 'prediction'}
            train_prediction = z['prediction'].copy()
        ty, tv = read_arrays(data, train_ids, 'target'), read_arrays(data, train_ids, 'valid')
        tb = data['geometry'][train_ids, 332:356].reshape(-1, 12, 2)
        cv, _ = distances(tb, ty, tv, data['scale'][train_ids])
        pe, _ = distances(train_prediction, ty, tv, data['scale'][train_ids])
        delta = cv-pe; expected = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0)))
        expected[~tv.all(1)] = np.nan
        np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-9, equal_nan=True)
        np.testing.assert_array_equal(pr['known'], tv.all(1))
        old = torch.load(ROOT/heads[key, 'bounded_fraction']['checkpoint'], map_location='cpu', weights_only=False)
        assert pr['hard_cut'] == old['preprocess']['hard_cut']
        assert pr['positive_easy_cut'] == old['preprocess']['positive_easy_cut']
        draws = None
        for arm in ARMS:
            rec = next(r for r in report['training'] if r['view'] == key and r['arm'] == arm)
            assert file_digest(ROOT/rec['checkpoint']) == rec['checkpoint_sha256']
            cp = torch.load(ROOT/rec['checkpoint'], map_location='cpu', weights_only=False)
            assert cp['identity']['identity'] == identity and cp['identity']['inputs_sha256'] == array_hash(train_ids, x, d)
            assert cp['identity']['labels_sha256'] == array_hash(y)
            assert cp['identity']['complete_mask_sha256'] == array_hash(pr['known'])
            assert cp['step'] == 3000 and cp['settings'] == cfg['training']
            for field in ('mean', 'std', 'known', 'weights', 'constant'):
                np.testing.assert_array_equal(cp['preprocess'][field], pr[field])
            assert cp['draws'][~pr['known']].sum() == 0 and cp['draws'].sum() == 3000*256
            if draws is None: draws = cp['draws']
            else: np.testing.assert_array_equal(draws, cp['draws'])
            fit_checks += 1
        cuts[key] = pr
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            ids, p = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        diff = p.astype(float)-b.astype(float)
        distance = np.sqrt(diff[..., 0]**2+diff[..., 1]**2).mean(1)*data['scale'][ids]
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        allowed = (distance > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
        rec = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT/rec['path']) == rec['sha256']
        with np.load(ROOT/rec['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            np.testing.assert_allclose(z['distance'], distance, rtol=1e-12, atol=1e-10)
            score = {f+'_'+a:z[f+'_'+a].copy() for f in ('frozen', 'refit') for a in ARMS}
            expected = expected_choices(score, allowed, ids)
            assert set(expected) == set(names)
            for name, bits in expected.items():
                np.testing.assert_array_equal(bits, z[name]); chosen[meta['seed']][name][ids] = bits
                choice_checks += 1
            scores[key] = score
        with np.load(ROOT/frozen[key]['path'], allow_pickle=False) as z:
            for arm in ARMS: np.testing.assert_array_equal(score['frozen_'+arm], z[arm])
            for name in expected:
                if name.startswith('frozen_'):
                    np.testing.assert_array_equal(expected[name], z[name.removeprefix('frozen_')])
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
        ids = np.flatnonzero(data['sites'] == meta['outer_site']); pr = cuts[key]; seed = meta['seed']
        masks['hard'][ids] = cv[ids] >= pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= pr['positive_easy_cut'])
        delta = cv[ids]-errors[seed][0][ids]
        cost = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0)))
        for name, score in scores[key].items():
            use = chosen[seed][name+'_strict_stop'][ids] & full[ids]
            r = next(r for r in report['conditional_quality'] if r['view'] == key and r['head'] == name)
            assert int(use.sum()) == r['selected_complete']
            np.testing.assert_allclose(((score[full[ids]]-cost[full[ids]])**2).mean(), r['complete_cost_MSE'])
            if use.any():
                np.testing.assert_allclose([score[use, 1].mean(), cost[use, 1].mean()],
                    [r['predicted_harm'], r['realized_harm']], rtol=1e-10, atol=1e-10)
            else: assert r['predicted_harm'] is None and r['realized_harm'] is None
    reductions = 0
    for name in names:
        ades, fdes = [], []
        for seed in cfg['seeds']:
            use = chosen[seed][name]
            ade, fde = np.where(use, errors[seed][0], cv), np.where(use, errors[seed][1], cf)
            ades.append(ade); fdes.append(fde); r = report['summaries'][name]['seeds'][str(seed)]
            assert int(use.sum()) == r['selected']
            assert int((use & ~valid.any(1)).sum()) == r['selected_unknown']
            assert int((use & ~full).sum()) == r['selected_incomplete']
            assert int((ade[masks['zero_CV']] > 0).sum()) == r['zero_CV_harmed']
            reductions += check_reduction(ade, cv, data['sites'], cfg['sites'], r['ADE'])
            reductions += check_reduction(fde, cf, data['sites'], cfg['sites'], r['FDE'])
            for group, mask in masks.items():
                reductions += check_reduction(ade[mask], cv[mask], data['sites'][mask], cfg['sites'], r['subsets'][group])
            for site in cfg['sites']:
                actual = [np.where(use, x, 0)[data['sites'] == site].mean() for x in bounds[seed]]
                np.testing.assert_allclose(actual, r['full_grid_absolute_gain_bounds'][site], rtol=1e-10, atol=1e-10)
        r = report['summaries'][name]
        reductions += check_reduction(np.mean(ades, 0), cv, data['sites'], cfg['sites'], r['ADE'])
        reductions += check_reduction(np.mean(fdes, 0), cf, data['sites'], cfg['sites'], r['FDE'])
        for group, mask in masks.items():
            reductions += check_reduction(np.mean(ades, 0)[mask], cv[mask], data['sites'][mask], cfg['sites'], r['subsets'][group])
    for arm in ARMS:
        for policy in cfg['policies']:
            a, b = [report['summaries'][f+'_'+arm+'_'+policy]['ADE']['by_scene'] for f in ('refit', 'frozen')]
            diff = np.array([a[s]['gain_percent']-b[s]['gain_percent'] for s in cfg['sites']])
            r = report['contrasts'][f'refit_minus_frozen_{arm}_{policy}']
            np.testing.assert_allclose(diff, r['scene_differences_pp'], atol=1e-10)
            rng = np.random.default_rng(r['seed']); indices = rng.integers(4, size=(r['resamples'], 4))
            np.testing.assert_allclose(np.quantile(diff[indices].mean(1), [.025, .975]), r['ci95_pp'], atol=1e-10)
    primary = report['summaries']['refit_bounded_fraction_strict_stop']
    gate = dict(exact_zero=all(v['zero_CV_harmed'] == 0 for v in primary['seeds'].values()),
        easy=all(-v['subsets']['positive_easy']['equal_scene_gain_percent'] <= 2 for v in primary['seeds'].values()),
        each_seed_positive_cv=all(v['ADE']['equal_scene_gain_percent'] > 0 for v in primary['seeds'].values()),
        positive_primary_ci=report['contrasts'][cfg['primary_contrast']]['ci95_pp'][0] > 0)
    assert gate == report['primary_gates'] and all(gate.values()) == report['primary_joint_empirical_pass']
    result = dict(analysis_sha256=file_digest(public/'analysis.json'), all_checks_passed=True,
        verifier_sha256=file_digest(Path(__file__)), fit_supervision_checks=fit_checks, policies_checked=choice_checks,
        scene_reductions=reductions, score_rows_checked=n*len(cfg['seeds'])*len(ARMS),
        independent_implementation_same_agent=True, independent_research_confirmation=False,
        preprocessing_recomputed_with_shared_helper=True, new_training=False, selection_changes=False, deployment=False)
    assert_current(identity); immutable_json(public/'independent_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
