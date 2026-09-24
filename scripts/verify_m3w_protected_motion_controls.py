"""Independent arithmetic, sampler and choice checks for protected motion controls."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_protected_motion_controls import load, folder
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.run_m3w_native_forecast import immutable_json, assert_current
from scripts.verify_m3w_native_joint_controls import distances
from src.data_unification.m3w_causal_recordings import BASELINES
from src.evaluation.m3w_experiment_contract import file_digest
import joblib
import numpy as np
import torch


def check_metrics(model, reference, sites, roster, expected):
    assert np.array_equal(np.isnan(model), np.isnan(reference))
    gains = []
    for site in roster:
        use = (sites == site) & np.isfinite(reference)
        row = expected['by_scene'][site]
        assert row['rows'] == int(use.sum())
        if not use.any():
            assert row['model_error'] is None
            continue
        mm, rr = model[use].mean(), reference[use].mean()
        np.testing.assert_allclose([mm, rr, mm-rr],
            [row['model_error'], row['reference_error'], row['absolute_harm']], atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(np.percentile(model[use], [95, 99]),
            [row['model_p95'], row['model_p99']], atol=1e-10, rtol=1e-12)
        if rr > 0:
            gain = 100*(1-mm/rr)
            np.testing.assert_allclose(gain, row['gain_percent'], atol=1e-10, rtol=1e-10)
            gains.append(gain)
        else:
            assert row['gain_percent'] is None
    if len(gains) == len(roster):
        np.testing.assert_allclose(np.mean(gains), expected['equal_scene_gain_percent'], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(min(gains), expected['worst_scene_gain_percent'], atol=1e-10, rtol=1e-10)
        if expected['bootstrap_resamples']:
            idx = np.random.default_rng(expected['bootstrap_seed']).integers(len(roster),
                size=(expected['bootstrap_resamples'], len(roster)))
            ci = np.percentile(np.asarray(gains)[idx].mean(1), [2.5, 97.5])
            np.testing.assert_allclose(ci, expected['scene_bootstrap_ci95'], atol=1e-10, rtol=1e-10)
        else:
            assert expected['scene_bootstrap_ci95'] is None
    else:
        assert expected['equal_scene_gain_percent'] is None
    return len(roster)


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, _, identity, refs = load()
    public, root = ROOT/cfg['reports'], ROOT/cfg['output']
    ap = public/'analysis.json'
    result = json.loads(ap.read_text())
    replay = json.loads((public/'verification.json').read_text())
    assert result['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(ap)
    receipt = json.loads((root/'decisions_complete.json').read_text())
    assert receipt['all_fits_complete'] and not receipt['target_arrays_loaded_for_decisions']
    n = len(data['sites'])
    y, mask = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    base = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = distances(base, y, mask, data['scale'])
    subsets = dict(complete=mask.all(1), zero_CV=mask.all(1) & (cv == 0),
                   hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    errors, endpoints, lower, upper, choices, matched_choices = {}, {}, {}, {}, {}, {}
    budgets, policies, matched_pairs = 0, 0, 0
    for seed in cfg['seeds']:
        for action in cfg['actions']:
            errors[seed, action], endpoints[seed, action] = np.full(n, np.nan), np.full(n, np.nan)
            lower[seed, action], upper[seed, action] = np.empty(n), np.empty(n)
            for head in cfg['heads']:
                for policy in cfg['policies']:
                    choices[seed, action, head, policy] = np.zeros(n, bool)
                if action != 'transformer':
                    for side in ('transformer', 'causal'):
                        matched_choices[seed, action, head, side] = np.zeros(n, bool)
    for key, meta in views.items():
        seed, held = meta['seed'], meta['outer_site']
        ids, train = np.flatnonzero(data['sites'] == held), np.flatnonzero(data['sites'] != held)
        ref = torch.load(ROOT/refs[key]['checkpoint'], map_location='cpu', weights_only=False)
        expected_known = mask[train].all(1)
        np.testing.assert_array_equal(ref['preprocess']['known'], expected_known)
        archives = next(r for r in result['decision_archives'] if r['view'] == key)
        assert file_digest(ROOT/archives['path']) == archives['sha256']
        with np.load(ROOT/archives['path'], allow_pickle=False) as z:
            stored = {k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(ids, stored['ids'])
        with np.load(ROOT/meta['outer_prediction']['prediction']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            neural = z['prediction'].copy()
        cut = receipt['cuts'][key]
        subsets['hard'][ids] = cv[ids] >= cut['hard_cut']
        subsets['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= cut['positive_easy_cut'])
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        for action in cfg['actions']:
            k = BASELINES.index(action) if action != 'transformer' else None
            p = neural if k is None else data['geometry'][ids, 308+24*k:332+24*k].reshape(-1, 12, 2)
            errors[seed, action][ids], endpoints[seed, action][ids] = distances(p, y[ids], mask[ids], data['scale'][ids])
            d = np.sqrt(np.sum((p.astype(float)-base[ids].astype(float))**2, axis=-1))*data['scale'][ids, None]
            observed = np.sqrt(np.sum((base[ids].astype(float)-y[ids].astype(float))**2, axis=-1))
            observed -= np.sqrt(np.sum((p.astype(float)-y[ids].astype(float))**2, axis=-1))
            observed = np.where(mask[ids], observed*data['scale'][ids, None], 0).sum(1)/12
            radius = np.where(mask[ids], 0, d).sum(1)/12
            lower[seed, action][ids], upper[seed, action][ids] = observed-radius, observed+radius
            for head in cfg['heads']:
                r = json.loads((folder(cfg, key, action, head)/'complete.json').read_text())
                assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
                cp = (torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
                      if head == 'neural' else joblib.load(ROOT/r['checkpoint']))
                np.testing.assert_array_equal(cp['draws'], ref['draws'])
                np.testing.assert_array_equal(cp['preprocess']['known'], expected_known)
                assert cp['draws'].sum() == cfg['training']['steps']*cfg['training']['batch_size']
                assert not cp['draws'][~expected_known].any()
                assert cp['preprocess']['training_sites'] == sorted(set(cfg['sites'])-{held})
                if head == 'neural':
                    assert cp['step'] == 3000 and cp['arm'] == 'bounded_fraction'
                else:
                    assert len(cp['model'].estimators_) == 128
                    train_pred = (None if action == 'transformer' else
                        data['geometry'][train, 308+24*k:332+24*k].reshape(-1, 12, 2))
                    if train_pred is None:
                        with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
                            np.testing.assert_array_equal(train, z['ids'])
                            train_pred = z['prediction'].copy()
                    has_distance = np.any(train_pred != base[train], axis=(1, 2))
                    np.testing.assert_array_equal(cp['sample_weight'], ref['draws']*has_distance)
                budgets += 1
                score = stored[action+'__'+head+'__score']
                support = (d.mean(1) > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
                assert np.isfinite(score).all() and (score >= 0).all()
                assert np.all(score.sum(1) <= d.mean(1)+2e-6*(1+d.mean(1)))
                net = support & (score[:, 0] > score[:, 1])
                strict = net & (score[:, 1] <= .1*score[:, 0])
                for policy, chosen in [('net_stop', net), ('strict_stop', strict)]:
                    np.testing.assert_array_equal(chosen, stored[action+'__'+head+'__'+policy])
                    choices[seed, action, head, policy][ids] = chosen
                    policies += 1
        for action in cfg['actions'][:-1]:
            for head in cfg['heads']:
                count = min(stored['transformer__'+head+'__strict_stop'].sum(), stored[action+'__'+head+'__strict_stop'].sum())
                for candidate, suffix in [('transformer', 'matched_transformer'), (action, 'matched_causal')]:
                    score = stored[candidate+'__'+head+'__score']
                    pool = np.flatnonzero(stored[candidate+'__'+head+'__strict_stop'])
                    ranked = sorted(pool, key=lambda i:(float(score[i, 1]-score[i, 0]), int(ids[i])))
                    selected = np.zeros(len(ids), bool); selected[ranked[:count]] = True
                    np.testing.assert_array_equal(selected, stored[action+'__'+head+'__'+suffix])
                    matched_choices[seed, action, head, suffix.removeprefix('matched_')][ids] = selected
                matched_pairs += 1
        print(json.dumps(dict(state='verified_view', view=key, budgets=budgets)), flush=True)
    reductions = 0
    for name, report in result['summaries'].items():
        action, head, policy = name.split('__')
        ades, fdes = [], []
        for seed in cfg['seeds']:
            chosen = np.ones(n, bool) if head == 'none' else choices[seed, action, head, policy]
            ade, fde = np.where(chosen, errors[seed, action], cv), np.where(chosen, endpoints[seed, action], cf)
            ades.append(ade); fdes.append(fde)
            r = report['seeds'][str(seed)]
            assert chosen.sum() == r['selected']
            assert (chosen & ~mask.any(1)).sum() == r['selected_unknown']
            assert (chosen & ~mask.all(1)).sum() == r['selected_incomplete']
            assert (ade[subsets['zero_CV']] > 0).sum() == r['zero_CV_harmed']
            reductions += check_metrics(ade, cv, data['sites'], cfg['sites'], r['ADE'])
            reductions += check_metrics(fde, cf, data['sites'], cfg['sites'], r['FDE'])
            for s in cfg['sites']:
                bounds = [np.where(chosen, b[seed, action], 0)[data['sites'] == s].mean() for b in (lower, upper)]
                np.testing.assert_allclose(bounds, r['full_grid_gain_bounds'][s], atol=1e-10, rtol=1e-10)
            for k, m in subsets.items():
                reductions += check_metrics(ade[m], cv[m], data['sites'][m], cfg['sites'], r['subsets'][k])
        reductions += check_metrics(np.mean(ades, 0), cv, data['sites'], cfg['sites'], report['ADE'])
        reductions += check_metrics(np.mean(fdes, 0), cf, data['sites'], cfg['sites'], report['FDE'])
        for k, m in subsets.items():
            reductions += check_metrics(np.mean(ades, 0)[m], cv[m], data['sites'][m], cfg['sites'], report['subsets'][k])
    matched_safety = {}
    for action in cfg['actions'][:-1]:
        for head in cfg['heads']:
            pair = {}
            for side in ('transformer', 'causal'):
                seed_rows = []
                for seed in cfg['seeds']:
                    use = matched_choices[seed, action, head, side]
                    a = 'transformer' if side == 'transformer' else action
                    e = np.where(use, errors[seed, a], cv)
                    sites = {}
                    for site in cfg['sites']:
                        stats = {}
                        for group, eligible in {'all':np.ones(n, bool), **subsets}.items():
                            m = eligible & (data['sites'] == site) & np.isfinite(cv)
                            ref = cv[m].mean() if m.any() else np.nan
                            value = e[m].mean() if m.any() else np.nan
                            stats[group] = dict(rows=int(m.sum()),
                                gain_percent=float(100*(1-value/ref)) if ref > 0 else None,
                                absolute_harm=float(value-ref) if m.any() else None,
                                harmed_rows=int((e[m] > cv[m]).sum()))
                        sites[site] = stats
                    seed_rows.append(dict(seed=seed, selected=int(use.sum()), by_scene=sites))
                easy = [s['positive_easy']['gain_percent'] for r in seed_rows for s in r['by_scene'].values()]
                pair[side] = dict(seeds=seed_rows,
                    worst_site_seed_easy_degradation_percent=max(0., -min(v for v in easy if v is not None)),
                    mean_equal_scene_ADE_gain_percent=float(np.mean([
                        s['all']['gain_percent'] for r in seed_rows for s in r['by_scene'].values()])),
                    mean_equal_scene_hard_gain_percent=float(np.mean([
                        s['hard']['gain_percent'] for r in seed_rows for s in r['by_scene'].values()])))
            matched_safety[action+'__'+head] = pair
    out = dict(all_checks_passed=True, analysis_sha256=file_digest(ap),
        verifier_sha256=file_digest(Path(__file__)), matched_fit_budgets_checked=budgets,
        policy_checks=policies, matched_pairs=matched_pairs, scene_reductions=reductions,
        matched_count_safety_supplement=matched_safety,
        independent_arithmetic_same_agent=True, independent_research_confirmation=False,
        external_readout=False, deployment=False)
    assert_current(identity)
    immutable_json(public/'independent_verification.json', out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
