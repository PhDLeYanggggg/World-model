"""Separate cost, weight, tree aggregation, decision and metric arithmetic checks."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_forest_cost import load, load_states
from scripts.run_m3w_temporal_intervention import training_arrays
from scripts.run_m3w_bounded_cost import features, read_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json, array_hash
from scripts.verify_m3w_temporal_intervention import manual_candidates
from scripts.verify_m3w_native_joint_controls import distances, check_reduction
from scripts.verify_m3w_tempered_cost import check_contrast
from src.world_model.m3w_forest_cost_head import ARMS, POLICIES
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def manual_choices(forest, neural, past, distance, ids):
    support = (distance > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
    def strict(p): return support & (p[:, 0] > p[:, 1]) & (p[:, 1] <= p[:, 0]/10.)
    f, nn = strict(forest), strict(neural)
    order = sorted(np.flatnonzero(support), key=lambda i: (float(neural[i, 1]-neural[i, 0]), int(ids[i])))
    matched = np.zeros(len(ids), bool); matched[order[:sum(f)]] = True
    return dict(forest=f, neural=nn, neural_matched=matched)


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, old, data, views, predictions, refs, previous, prior, identity = load()
    records, states = load_states(cfg, views, identity); public = ROOT/cfg['reports']
    report = json.loads((public/'analysis.json').read_text()); replay = json.loads((public/'replay.json').read_text())
    assert report['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(public/'analysis.json')
    assert report['decision_manifest_sha256'] == file_digest(ROOT/cfg['output']/'decisions_complete.json')
    n = len(data['sites']); chosen = {s: {p: np.zeros(n, bool) for p in POLICIES} for s in cfg['seeds']}
    forecasts = {s: {a: np.empty((n, 12, 2)) for a in ARMS} for s in cfg['seeds']}
    fit_rows = choices = tree_score_rows = 0
    for key, meta in views.items():
        for arm in ARMS:
            ids, x, y, d, pr, bits, w, q = training_arrays(meta, data, refs, key, old, arm)
            cp = states[key, arm]; known = pr['known']
            with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
                np.testing.assert_array_equal(ids, z['ids']); raw = z['prediction'].copy()
            base = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
            alternative, _ = manual_candidates(base, raw)
            target, valid = read_arrays(data, ids, 'target'), read_arrays(data, ids, 'valid')
            bc, _ = distances(base, target, valid, data['scale'][ids])
            nc, _ = distances(alternative[arm], target, valid, data['scale'][ids])
            yy = np.column_stack((np.maximum(bc-nc, 0), np.maximum(nc-bc, 0))); yy[~valid.all(1)] = np.nan
            np.testing.assert_allclose(yy, y, rtol=1e-10, atol=1e-8, equal_nan=True)
            np.testing.assert_array_equal(cp['draws'], previous[key, arm]['draws'])
            ww = cp['draws'].astype(float)*w*d/pr['cost_scale']; ww /= ww[ww > 0].mean()
            np.testing.assert_array_equal(ww, cp['sample_weight'])
            assert not ww[~known].any()
            for f in ('mean', 'std', 'known'): np.testing.assert_array_equal(cp['preprocess'][f], pr[f])
            assert cp['identity']['inputs_sha256'] == array_hash(ids, x, d, q)
            assert cp['identity']['labels_sha256'] == array_hash(y)
            model = cp['model']
            assert model.get_params()['bootstrap'] is False
            assert model.get_params()['min_samples_leaf'] == cfg['forest']['min_samples_leaf']
            assert len(model.estimators_) == cfg['forest']['trees']
            for tree in model.estimators_:
                leaf = tree.tree_.children_left == -1
                values = tree.tree_.value[leaf, :, 0]
                assert (values >= 0).all() and (values.sum(1) <= 1+1e-12).all()
            fit_rows += len(ids)
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            hi, raw = z['ids'].copy(), z['prediction'].copy()
        b = data['geometry'][hi, 332:356].reshape(-1, 12, 2); q, _ = manual_candidates(b, raw)
        archive = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT/archive['path']) == archive['sha256']
        with np.load(ROOT/archive['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(hi, z['ids'])
            for arm in ARMS:
                cp = states[key, arm]; pr = cp['preprocess']; model = cp['model']
                hx, _, _ = features(data['geometry'][hi], q[arm], data['scale'][hi])
                standardized = ((hx.astype(float)-pr['mean'])/pr['std']).astype(np.float32)
                fraction = np.zeros((len(hi), 2))
                for tree in model.estimators_: fraction += tree.predict(standardized)
                fraction /= len(model.estimators_)
                dd = np.sqrt(np.square(q[arm]-b).sum(-1)).mean(1)*data['scale'][hi]
                np.testing.assert_allclose(dd, z[arm+'_distance'], rtol=1e-10, atol=1e-8)
                # Alternate algebra may introduce only roundoff, not a changed leaf or decision.
                np.testing.assert_allclose(fraction*dd[:, None], z[arm+'_forest_score'], rtol=1e-10, atol=1e-8)
                masks = manual_choices(z[arm+'_forest_score'], z[arm+'_neural_score'],
                    data['geometry'][hi, :16].reshape(-1, 8, 2), z[arm+'_distance'], hi)
                for name, use in masks.items():
                    name = arm+'_'+name; np.testing.assert_array_equal(use, z[name])
                    chosen[meta['seed']][name][hi] = use; choices += 1
                forecasts[meta['seed']][arm][hi] = q[arm]; tree_score_rows += len(hi)
        print(json.dumps(dict(view=key, state='separately_checked')), flush=True)
    y, v = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    b = data['geometry'][:, 332:356].reshape(n, 12, 2); cv, cf = distances(b, y, v, data['scale']); full = v.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    for key, meta in views.items():
        use = data['sites'] == meta['outer_site']; pr = states[key, 'ramp']['preprocess']
        masks['hard'][use] = cv[use] >= pr['hard_cut']
        masks['positive_easy'][use] = (cv[use] > 0) & (cv[use] <= pr['positive_easy_cut'])
    errors, bounds = {}, {}
    safe = np.where(v[..., None], y, 0).astype(float)
    be = np.sqrt(np.square(b.astype(float)-safe).sum(-1))*data['scale'][:, None]
    for seed in cfg['seeds']:
        for arm in ARMS:
            p = forecasts[seed][arm]; errors[seed, arm] = distances(p, y, v, data['scale'])
            pe = np.sqrt(np.square(p-safe).sum(-1))*data['scale'][:, None]
            dd = np.sqrt(np.square(p-b).sum(-1))*data['scale'][:, None]
            observed = np.where(v, be-pe, 0).sum(1)/12
            unknown = np.where(v, 0, dd).sum(1)/12
            bounds[seed, arm] = observed-unknown, observed+unknown
    reductions = 0
    for name in POLICIES:
        arm = name.split('_')[0]; ades, fdes = [], []
        for seed in cfg['seeds']:
            use = chosen[seed][name]
            ade, fde = np.where(use, errors[seed, arm][0], cv), np.where(use, errors[seed, arm][1], cf)
            ades.append(ade); fdes.append(fde); r = report['summaries'][name]['seeds'][str(seed)]
            assert r['selected'] == int(use.sum()) and r['selected_unknown'] == int((use & ~v.any(1)).sum())
            assert r['selected_incomplete'] == int((use & ~full).sum())
            assert r['zero_CV_harmed'] == int((ade[masks['zero_CV']] > 0).sum())
            reductions += check_reduction(ade, cv, data['sites'], cfg['sites'], r['ADE'])
            reductions += check_reduction(fde, cf, data['sites'], cfg['sites'], r['FDE'])
            for g, m in masks.items():
                reductions += check_reduction(ade[m], cv[m], data['sites'][m], cfg['sites'], r['subsets'][g])
            for site in cfg['sites']:
                np.testing.assert_allclose([np.where(use, q, 0)[data['sites']==site].mean() for q in bounds[seed, arm]],
                    r['full_grid_absolute_gain_bounds'][site], rtol=1e-10, atol=1e-8)
        r = report['summaries'][name]
        reductions += check_reduction(np.mean(ades, 0), cv, data['sites'], cfg['sites'], r['ADE'])
        reductions += check_reduction(np.mean(fdes, 0), cf, data['sites'], cfg['sites'], r['FDE'])
        for g, m in masks.items():
            reductions += check_reduction(np.mean(ades, 0)[m], cv[m], data['sites'][m], cfg['sites'], r['subsets'][g])
    for arm in ARMS:
        for control, c in report['contrasts'][arm].items():
            check_contrast(report['summaries'][arm+'_forest']['ADE']['by_scene'],
                           report['summaries'][arm+'_'+control]['ADE']['by_scene'], cfg['sites'], c)
    r = report['summaries']['ramp_forest']; contrast = report['contrasts']['ramp']['neural']
    checks = dict(positive_primary_ci=contrast['ci95_pp'][0] > 0,
        each_seed_positive_cv=all(s['ADE']['equal_scene_gain_percent'] > 0 for s in r['seeds'].values()),
        aggregate_easy=all(s['subsets']['positive_easy']['equal_scene_gain_percent'] >= -2 for s in r['seeds'].values()),
        each_scene_seed_easy=all(s['gain_percent'] >= -2 for seed in r['seeds'].values()
            for s in seed['subsets']['positive_easy']['by_scene'].values()),
        exact_zero=not any(s['zero_CV_harmed'] for s in r['seeds'].values()))
    assert checks == report['primary_gates'] and all(checks.values()) == report['primary_joint_empirical_pass']
    assert_current(identity)
    immutable_json(public/'separate_verification.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
        verifier_sha256=file_digest(Path(__file__)), all_checks_passed=True,
        fitting_arm_row_instances=fit_rows, separately_summed_tree_score_rows=tree_score_rows,
        policy_choices_checked=choices, scene_reductions_checked=reductions, bounds_and_gates_checked=True,
        context_smoothness_and_conditional_risk_replayed_only=True,
        same_agent_shared_source_arrays=True, independent_research_confirmation=False))
    print(json.dumps(dict(verified=True, fitting_arm_rows=fit_rows, choices=choices, reductions=reductions)))


if __name__ == '__main__': main()
