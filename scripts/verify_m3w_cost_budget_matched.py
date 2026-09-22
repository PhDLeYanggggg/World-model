"""Separate arithmetic verification of fixed equal-budget objective controls."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_cost_budget_matched import load, load_states, ARMS
import numpy as np
import torch
from scripts.run_m3w_bounded_cost import read_arrays, training_data
from scripts.run_m3w_native_forecast import assert_current, immutable_json, array_hash
from scripts.verify_m3w_native_joint_controls import distances, check_reduction
from scripts.verify_m3w_tempered_cost import expected_choices, check_contrast
from src.evaluation.m3w_experiment_contract import file_digest


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg, data, views, predictions, controls, prior, refs, identity = load()
    records, states = load_states(cfg, views, refs, identity)
    public = ROOT/cfg['reports']
    report = json.loads((public/'analysis.json').read_text())
    replay = json.loads((public/'replay.json').read_text())
    assert report['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(public/'analysis.json')
    assert report['decision_manifest_sha256'] == file_digest(ROOT/cfg['output']/'decisions_complete.json')
    n = len(data['sites'])
    policies = ('net_stop', 'strict_stop', 'matched_count')
    chosen = {s:{a:{p:np.zeros(n, bool) for p in policies} for a in ARMS} for s in cfg['seeds']}
    predictions_by_seed = {s:np.empty((n, 12, 2), np.float32) for s in cfg['seeds']}
    anchors = {r['view']:r for r in controls['archives']}
    checks, fit_rows = 0, 0
    for key, meta in views.items():
        train_ids, x, labels, d, pr = training_data(meta, data)
        np.testing.assert_array_equal(train_ids, np.flatnonzero(data['sites'] != meta['outer_site']))
        with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
            assert set(z.files) == {'ids', 'prediction'}
            np.testing.assert_array_equal(z['ids'], train_ids)
            train_p = z['prediction'].copy()
        ty, tv = read_arrays(data, train_ids, 'target'), read_arrays(data, train_ids, 'valid')
        tb = data['geometry'][train_ids, 332:356].reshape(-1, 12, 2)
        bc, _ = distances(tb, ty, tv, data['scale'][train_ids])
        ec, _ = distances(train_p, ty, tv, data['scale'][train_ids])
        expected = np.column_stack((np.maximum(bc-ec, 0), np.maximum(ec-bc, 0)))
        expected[~tv.all(1)] = np.nan
        np.testing.assert_allclose(expected, labels, rtol=1e-12, atol=1e-9, equal_nan=True)
        np.testing.assert_array_equal(pr['known'], tv.all(1))
        fit_rows += len(train_ids)
        for arm in cfg['training_arms']:
            cp = states[key, arm]
            assert cp['identity']['inputs_sha256'] == array_hash(train_ids, x, d)
            assert cp['identity']['labels_sha256'] == array_hash(labels)
            assert cp['identity']['complete_mask_sha256'] == array_hash(pr['known'])
            for f in ('mean', 'std', 'known', 'weights', 'constant'):
                np.testing.assert_array_equal(cp['preprocess'][f], pr[f])
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            ids, prediction = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        diff = prediction.astype(float)-b.astype(float)
        distance = np.sqrt(diff[..., 0]**2+diff[..., 1]**2).mean(1)*data['scale'][ids]
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        allowed = (distance > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
        with np.load(ROOT/anchors[key]['path'], allow_pickle=False) as z:
            score = z['frozen_bounded_fraction']
            anchor = allowed & (score[:, 0] > score[:, 1]) & (score[:, 1] <= .1*score[:, 0])
            np.testing.assert_array_equal(anchor, z[cfg['matched_count_reference']])
        archive = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT/archive['path']) == archive['sha256']
        with np.load(ROOT/archive['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            np.testing.assert_array_equal(z['anchor'], anchor)
            np.testing.assert_allclose(z['distance'], distance, rtol=1e-12, atol=1e-10)
            for arm in ARMS:
                for policy, bits in expected_choices(z[arm+'_score'], allowed, ids, anchor).items():
                    np.testing.assert_array_equal(bits, z[arm+'_'+policy])
                    chosen[meta['seed']][arm][policy][ids] = bits
                    checks += 1
        predictions_by_seed[meta['seed']][ids] = prediction
    y, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    baseline = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = distances(baseline, y, valid, data['scale'])
    full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    for key, meta in views.items():
        ids = np.flatnonzero(data['sites'] == meta['outer_site'])
        pr = states[key, 'tempered']['preprocess']
        masks['hard'][ids] = cv[ids] >= pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= pr['positive_easy_cut'])
    errors, bounds = {}, {}
    for seed, prediction in predictions_by_seed.items():
        errors[seed] = distances(prediction, y, valid, data['scale'])
        safe = np.where(valid[..., None], y, 0).astype(float)
        dd = np.sqrt(np.sum((prediction.astype(float)-baseline.astype(float))**2, axis=-1))*data['scale'][:, None]
        pe = np.sqrt(np.sum((prediction.astype(float)-safe)**2, axis=-1))*data['scale'][:, None]
        be = np.sqrt(np.sum((baseline.astype(float)-safe)**2, axis=-1))*data['scale'][:, None]
        observed = np.where(valid, be-pe, 0).sum(1)/12
        unknown = np.where(valid, 0, dd).sum(1)/12
        bounds[seed] = observed-unknown, observed+unknown
    reductions = 0
    for arm in ARMS:
        for policy in policies:
            ades, fdes = [], []
            for seed in cfg['seeds']:
                use = chosen[seed][arm][policy]
                ade, fde = np.where(use, errors[seed][0], cv), np.where(use, errors[seed][1], cf)
                ades.append(ade)
                fdes.append(fde)
                r = report['summaries'][arm][policy]['seeds'][str(seed)]
                assert int(use.sum()) == r['selected'] and int((use & ~valid.any(1)).sum()) == r['selected_unknown']
                assert int((use & ~full).sum()) == r['selected_incomplete']
                assert int((ade[masks['zero_CV']] > 0).sum()) == r['zero_CV_harmed']
                reductions += check_reduction(ade, cv, data['sites'], cfg['sites'], r['ADE'])
                reductions += check_reduction(fde, cf, data['sites'], cfg['sites'], r['FDE'])
                for name, mask in masks.items():
                    reductions += check_reduction(ade[mask], cv[mask], data['sites'][mask], cfg['sites'], r['subsets'][name])
                for site in cfg['sites']:
                    np.testing.assert_allclose([np.where(use, v, 0)[data['sites']==site].mean() for v in bounds[seed]],
                        r['full_grid_absolute_gain_bounds'][site], rtol=1e-10, atol=1e-10)
            r = report['summaries'][arm][policy]
            reductions += check_reduction(np.mean(ades, 0), cv, data['sites'], cfg['sites'], r['ADE'])
            reductions += check_reduction(np.mean(fdes, 0), cf, data['sites'], cfg['sites'], r['FDE'])
            for name, mask in masks.items():
                reductions += check_reduction(np.mean(ades, 0)[mask], cv[mask], data['sites'][mask], cfg['sites'], r['subsets'][name])
    for policy in policies:
        for arm in cfg['training_arms']:
            check_contrast(report['summaries']['tempered'][policy]['ADE']['by_scene'],
                report['summaries'][arm][policy]['ADE']['by_scene'], cfg['sites'],
                report['contrasts'][policy]['tempered_minus_'+arm])
    p = report['summaries']['tempered']['strict_stop']
    c = report['contrasts']['strict_stop']
    gates = dict(exact_zero=all(r['zero_CV_harmed'] == 0 for r in p['seeds'].values()),
        easy=all(-r['subsets']['positive_easy']['equal_scene_gain_percent'] <= 2 for r in p['seeds'].values()),
        each_seed_positive_cv=all(r['ADE']['equal_scene_gain_percent'] > 0 for r in p['seeds'].values()),
        positive_vs_native_ci=c['tempered_minus_native']['ci95_pp'][0] > 0,
        positive_vs_fraction_ci=c['tempered_minus_fraction']['ci95_pp'][0] > 0)
    assert gates == report['primary_gates'] and all(gates.values()) == report['primary_joint_empirical_pass']
    assert report['summaries']['tempered'] == prior['summaries']['wide_long']
    assert sum(r['fit']['step'] for r in records.values()) == report['new_updates'] == 288000
    assert sum(r['fit']['total_draws'] for r in records.values()) == report['new_draws'] == 73728000
    result = dict(analysis_sha256=file_digest(public/'analysis.json'), verifier_sha256=file_digest(Path(__file__)),
        all_checks_passed=True, policy_choices_checked=checks, scene_reductions=reductions,
        supervision_rows_checked=fit_rows, endpoint_states_checked=36,
        cached_reference_summary_exact=True, independent_implementation_same_agent=True,
        fitting_diagnostics_replayed_by_runner_not_separate_implementation=True,
        preprocessing_recomputed_with_shared_helper=True, independent_research_confirmation=False, new_training=False)
    assert_current(identity)
    immutable_json(public/'independent_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
