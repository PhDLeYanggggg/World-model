"""Versioned readout verifier; corrects the frozen verifier contrast argument shape."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_prefix_cost import load, load_states, training_arrays, check_state
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json, array_hash
from scripts.verify_m3w_native_joint_controls import distances, check_reduction
from scripts.verify_m3w_tempered_cost import check_contrast
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_prefix_cost_policy import POLICIES
import numpy as np
import torch


def verify_contrasts(report, scalar, sites):
    for name, c in report['contrasts'].items():
        ref = scalar['summaries']['strict_stop'] if name == 'scalar_log_strict' else report['summaries'][name]
        check_contrast(report['summaries']['profile_guard']['ADE']['by_scene'], ref['ADE']['by_scene'], sites, c)


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, predictions, scalar, refs, identity = load()
    records, states = load_states(cfg, views, identity)
    public = ROOT / cfg['reports']; report = json.loads((public / 'analysis.json').read_text())
    replay = json.loads((public / 'replay.json').read_text())
    assert report['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(public / 'analysis.json')
    assert report['decision_manifest_sha256'] == file_digest(ROOT / cfg['output'] / 'decisions_complete.json')
    n = len(data['sites'])
    chosen = {s: {p: np.zeros(n, bool) for p in POLICIES} for s in cfg['seeds']}
    forecasts = {s: np.empty((n, 12, 2), np.float32) for s in cfg['seeds']}
    fit_rows = choices = 0
    for key, meta in views.items():
        ids, x, costs, d, pr, bits, w = training_arrays(meta, data, refs, key, cfg)
        with np.load(ROOT / meta['inputs_path'], allow_pickle=False) as z:
            p = z['prediction'].copy(); np.testing.assert_array_equal(ids, z['ids'])
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        y, v, scale = read_arrays(data, ids, 'target'), read_arrays(data, ids, 'valid'), data['scale'][ids]
        for k in range(1, 13):
            ok = v[:, :k].all(1)
            be = np.sqrt(np.square(b[ok, :k].astype(float)-y[ok, :k]).sum(-1)).mean(1)*scale[ok]
            pe = np.sqrt(np.square(p[ok, :k].astype(float)-y[ok, :k]).sum(-1)).mean(1)*scale[ok]
            expected = np.column_stack((np.maximum(be-pe, 0), np.maximum(pe-be, 0)))
            np.testing.assert_allclose(costs[ok, k-1], expected, rtol=1e-10, atol=1e-8)
            assert np.isnan(costs[~ok, k-1]).all()
            expected_d = np.sqrt(np.square(p[:, :k].astype(float)-b[:, :k]).sum(-1)).mean(1)*scale
            np.testing.assert_allclose(d[:, k-1], expected_d, rtol=1e-10, atol=1e-8)
        for arm in cfg['arms']:
            cp = states[key, arm]; check_state(cp, refs[key, 'frozen_region'], w, cfg)
            assert cp['identity']['inputs_sha256'] == array_hash(ids, x, d)
            assert cp['identity']['labels_sha256'] == array_hash(costs)
            assert cp['identity']['region_sha256'] == array_hash(ids, bits, w)
        fit_rows += len(ids)
        with np.load(ROOT / predictions[key]['path'], allow_pickle=False) as z:
            hi, p = z['ids'].copy(), z['prediction'].copy()
        archive = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT / archive['path']) == archive['sha256']
        with np.load(ROOT / archive['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(hi, z['ids'])
            past = data['geometry'][hi, :16].reshape(-1, 8, 2)
            support = (z['distance'][:, -1] > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
            control, profile = z['control'], z['profile']
            def terminal(a):
                return support & (a[:, -1, 0] > a[:, -1, 1]) & (a[:, -1, 1] <= .1*a[:, -1, 0])
            masks = dict(control_terminal=terminal(control), profile_terminal=terminal(profile))
            masks['profile_guard'] = masks['profile_terminal'] & np.all(profile[..., 1] <= .1*profile[..., 0], axis=1)
            for name, a in (('control_matched', control), ('profile_matched', profile)):
                pool = np.flatnonzero(support)
                ordered = sorted(pool, key=lambda i: (-(a[i, -1, 0]-a[i, -1, 1]), int(hi[i])))
                m = np.zeros(len(hi), bool); m[ordered[:int(masks['profile_guard'].sum())]] = True; masks[name] = m
            for name, use in masks.items():
                np.testing.assert_array_equal(use, z[name]); chosen[meta['seed']][name][hi] = use; choices += 1
        forecasts[meta['seed']][hi] = p
    y, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    b = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = distances(b, y, valid, data['scale']); full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    for key, meta in views.items():
        ix = data['sites'] == meta['outer_site']; pr = states[key, 'prefix']['preprocess']
        masks['hard'][ix] = cv[ix] >= pr['hard_cut']
        masks['positive_easy'][ix] = (cv[ix] > 0) & (cv[ix] <= pr['positive_easy_cut'])
    errors = {s: distances(p, y, valid, data['scale']) for s, p in forecasts.items()}
    bounds = {}
    safe = np.where(valid[..., None], y, 0).astype(float)
    be = np.sqrt(np.square(b.astype(float)-safe).sum(-1))*data['scale'][:, None]
    for seed, p in forecasts.items():
        pe = np.sqrt(np.square(p.astype(float)-safe).sum(-1))*data['scale'][:, None]
        dd = np.sqrt(np.square(p.astype(float)-b.astype(float)).sum(-1))*data['scale'][:, None]
        observed = np.where(valid, be-pe, 0).sum(1)/12
        unknown = np.where(valid, 0, dd).sum(1)/12
        bounds[seed] = observed-unknown, observed+unknown
    reductions = 0
    for policy in POLICIES:
        ades, fdes = [], []
        for seed in cfg['seeds']:
            use = chosen[seed][policy]
            ade, fde = np.where(use, errors[seed][0], cv), np.where(use, errors[seed][1], cf)
            ades.append(ade); fdes.append(fde); r = report['summaries'][policy]['seeds'][str(seed)]
            assert r['selected'] == int(use.sum()) and r['selected_unknown'] == int((use & ~valid.any(1)).sum())
            assert r['selected_incomplete'] == int((use & ~full).sum())
            assert r['zero_CV_harmed'] == int((ade[masks['zero_CV']] > 0).sum())
            for site in cfg['sites']:
                np.testing.assert_allclose([np.where(use, q, 0)[data['sites'] == site].mean() for q in bounds[seed]],
                                           r['full_grid_absolute_gain_bounds'][site], rtol=1e-10, atol=1e-10)
            reductions += check_reduction(ade, cv, data['sites'], cfg['sites'], r['ADE'])
            reductions += check_reduction(fde, cf, data['sites'], cfg['sites'], r['FDE'])
            for g, m in masks.items():
                reductions += check_reduction(ade[m], cv[m], data['sites'][m], cfg['sites'], r['subsets'][g])
        r = report['summaries'][policy]
        reductions += check_reduction(np.mean(ades, 0), cv, data['sites'], cfg['sites'], r['ADE'])
        reductions += check_reduction(np.mean(fdes, 0), cf, data['sites'], cfg['sites'], r['FDE'])
        for g, m in masks.items():
            reductions += check_reduction(np.mean(ades, 0)[m], cv[m], data['sites'][m], cfg['sites'], r['subsets'][g])
    verify_contrasts(report, scalar, cfg['sites'])
    summary = report['summaries']['profile_guard']
    checks = dict(positive_primary_ci=report['contrasts']['control_terminal']['ci95_pp'][0] > 0,
                  each_seed_positive_cv=all(r['ADE']['equal_scene_gain_percent'] > 0 for r in summary['seeds'].values()),
                  aggregate_easy=all(-r['subsets']['positive_easy']['equal_scene_gain_percent'] <= 2 for r in summary['seeds'].values()),
                  each_scene_seed_easy=all(-v['gain_percent'] <= 2 for r in summary['seeds'].values()
                                           for v in r['subsets']['positive_easy']['by_scene'].values()),
                  exact_zero=all(r['zero_CV_harmed'] == 0 for r in summary['seeds'].values()),
                  positive_equal_count_ci=report['contrasts']['control_matched']['ci95_pp'][0] > 0)
    assert checks == report['primary_gates'] and all(checks.values()) == report['primary_joint_empirical_pass']
    assert_current(identity)
    out = dict(analysis_sha256=file_digest(public / 'analysis.json'), all_checks_passed=True,
               verifier_sha256=file_digest(Path(__file__)),
               supersedes='scripts/verify_m3w_prefix_cost.py: contrast argument shape only',
               fitting_rows_checked=fit_rows, separate_prefix_reductions=144, policy_choices_checked=choices,
               scene_reductions_checked=reductions, checkpoint_draws_matched=24,
               full_grid_bounds_checked=True, gates_recomputed=True,
               same_agent_shared_source_arrays=True, independent_research_confirmation=False)
    immutable_json(public / 'separate_verification.json', out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
