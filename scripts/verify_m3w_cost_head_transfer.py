"""Separate choice, error and interval arithmetic for frozen cost-head transfer."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_m3w_cost_head_transfer import load, read_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from scripts.verify_m3w_native_joint_controls import distances, check_reduction
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_bounded_cost_head import ARMS


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, _, heads, predictions, identity = load()
    public = ROOT/cfg['reports']; report = json.loads((public/'analysis.json').read_text())
    replay = json.loads((public/'replay.json').read_text())
    assert report['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(public/'analysis.json')
    assert file_digest(ROOT/cfg['output']/'decisions_complete.json') == report['decision_manifest_sha256']
    n = len(data['sites']); names = list(report['summaries'])
    chosen = {s:{k:np.zeros(n, bool) for k in names} for s in cfg['seeds']}
    pred = {s:np.empty((n, 12, 2), np.float32) for s in cfg['seeds']}
    dist = {s:np.empty(n) for s in cfg['seeds']}
    cuts, stored_scores, choice_checks = {}, {}, 0
    for key, meta in views.items():
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            ids, p = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        delta = p.astype(float)-b.astype(float)
        d = np.sqrt(delta[..., 0]**2+delta[..., 1]**2).mean(1)*data['scale'][ids]
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        allowed = (d > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
        rec = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT/rec['path']) == rec['sha256']
        with np.load(ROOT/rec['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            np.testing.assert_allclose(z['distance'], d, rtol=1e-12, atol=1e-10)
            expected = dict(floor=np.zeros(len(ids), bool), uncontrolled=np.ones(len(ids), bool), past_stop=allowed)
            for arm in ARMS:
                s = z[arm].copy(); stored_scores[key, arm] = s
                assert np.isfinite(s).all() and (s >= 0).all()
                expected[arm+'_net_stop'] = allowed & (s[:, 0] > s[:, 1])
                expected[arm+'_strict_stop'] = expected[arm+'_net_stop'] & (s[:, 1] <= .1*s[:, 0])
            count = int(expected['bounded_fraction_strict_stop'].sum())
            for arm in ARMS:
                s = stored_scores[key, arm]
                order = sorted(np.flatnonzero(allowed), key=lambda i:(float(s[i, 1]-s[i, 0]), int(ids[i])))
                bits = np.zeros(len(ids), bool); bits[order[:count]] = True
                expected[arm+'_matched_count'] = bits
            assert set(expected) == set(names)
            for name, bits in expected.items():
                np.testing.assert_array_equal(bits, z[name]); chosen[meta['seed']][name][ids] = bits
                choice_checks += 1
        cp = torch.load(ROOT/heads[key, 'direct_native']['checkpoint'], map_location='cpu', weights_only=False)
        assert meta['outer_site'] not in cp['preprocess']['training_sites']
        cuts[key] = cp['preprocess']
        pred[meta['seed']][ids], dist[meta['seed']][ids] = p, d
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
        harm = np.maximum(errors[meta['seed']][0][ids]-cv[ids], 0)
        for arm in ARMS:
            use = chosen[meta['seed']][arm+'_strict_stop'][ids] & full[ids]
            r = next(r for r in report['conditional_quality'] if r['view'] == key and r['arm'] == arm)
            assert int(use.sum()) == r['selected_complete']
            if use.any():
                np.testing.assert_allclose([stored_scores[key, arm][use, 1].mean(), harm[use].mean()],
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
            assert int((use & (dist[seed] > 0)).sum()) == r['effective_switches']
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
    for policy in ('strict_stop', 'matched_count'):
        for right in ('direct_native', 'bounded_native'):
            a, b = [report['summaries'][name+'_'+policy]['ADE']['by_scene'] for name in ('bounded_fraction', right)]
            diff = np.array([a[s]['gain_percent']-b[s]['gain_percent'] for s in cfg['sites']])
            r = report['contrasts']['bounded_fraction_minus_'+right+'_'+policy]
            np.testing.assert_allclose(diff, r['scene_differences_pp'], atol=1e-10)
            rng = np.random.default_rng(r['seed']); indices = rng.integers(4, size=(r['resamples'], 4))
            np.testing.assert_allclose(np.quantile(diff[indices].mean(1), [.025, .975]), r['ci95_pp'], atol=1e-10)
    result = dict(analysis_sha256=file_digest(public/'analysis.json'), all_checks_passed=True,
        verifier_sha256=file_digest(Path(__file__)), policies_checked=choice_checks,
        scene_reductions=reductions, score_rows_checked=n*len(cfg['seeds'])*len(ARMS),
        independent_implementation_same_agent=True, independent_research_confirmation=False,
        new_training=False, selection_changes=False, deployment=False)
    assert_current(identity)
    immutable_json(public/'independent_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
