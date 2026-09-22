"""Separate target/objective, exact exposure, choices and metric verification."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_fraction_square import load, load_states
from scripts.run_m3w_temporal_intervention import training_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json, array_hash
from scripts.run_m3w_bounded_cost import read_arrays, features
from scripts.verify_m3w_temporal_intervention import manual_candidates
from scripts.verify_m3w_native_joint_controls import distances, check_reduction
from scripts.verify_m3w_tempered_cost import check_contrast
from src.world_model.m3w_native_gain_harm import standardized
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_fraction_square_eval import POLICIES
import numpy as np
import torch


def fractions_from_logits(logits):
    p = np.logaddexp(0., np.asarray(logits, float))
    return p/(1+p.sum(1, keepdims=True))


def predict_fractions(cp, x):
    pars = cp['model']; out = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            z = torch.from_numpy(standardized(x[start:start+4096], cp['preprocess']))
            h = torch.nn.functional.gelu(torch.nn.functional.linear(z, pars['network.0.weight'], pars['network.0.bias']))
            logits = torch.nn.functional.linear(h, pars['network.2.weight'], pars['network.2.bias'])
            out.append(fractions_from_logits(logits.numpy()))
    return np.concatenate(out)


def native_scores(cp, x, distance):
    """Assemble the saved forward pass without its model class; retain float32 operation order."""
    pars, pr, out = cp['model'], cp['preprocess'], []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            z = torch.from_numpy(standardized(x[start:start+4096], pr))
            d = torch.from_numpy((distance[start:start+4096]/pr['cost_scale']).astype(np.float32))
            h = torch.nn.functional.gelu(torch.nn.functional.linear(z, pars['network.0.weight'], pars['network.0.bias']))
            positive = torch.nn.functional.softplus(torch.nn.functional.linear(h, pars['network.2.weight'], pars['network.2.bias']))
            q = d[:, None]*positive/(1+positive.sum(1, keepdim=True))
            q = torch.where(d[:, None] == 0, 0., q)
            out.append(q.numpy().astype(float)*pr['cost_scale'])
    return np.concatenate(out)


def empirical_square(prediction, target, weight):
    p, q, w = map(np.asarray, (prediction, target, weight))
    if (p.shape != q.shape or p.ndim != 2 or p.shape[1] != 2 or w.shape != (len(p),)
            or not np.isfinite(p).all() or not np.isfinite(q).all() or not np.isfinite(w).all()
            or (w < 0).any() or not w.sum() > 0):
        raise ValueError('Finite aligned fractions and positive empirical fitting mass required')
    return float(np.sum(w*np.square(p-q).sum(1))/(2*np.sum(w)))


def manual_choices(score, eligible, ids, count):
    strict = np.zeros(len(ids), bool); pool = list(np.flatnonzero(eligible))
    for i in pool:
        g, h = map(float, score[i]); strict[i] = g > h and h <= .1*g
    def rank(ratio):
        def key(i):
            g, h = map(float, score[i])
            return (h/g if g > 0 else float('inf')) if ratio else h-g, int(ids[i])
        order = sorted(pool, key=key)
        out = np.zeros(len(ids), bool); out[order[:count]] = True; return out
    return dict(square_strict=strict, square_ratio=rank(True), square_gain=rank(False))


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, old, data, views, predictions, refs, neural, forest, ranking, identity = load()
    records, states = load_states(cfg, views, identity); public = ROOT/cfg['reports']
    report = json.loads((public/'analysis.json').read_text()); replay = json.loads((public/'replay.json').read_text())
    assert report['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(public/'analysis.json')
    assert report['decision_manifest_sha256'] == file_digest(ROOT/cfg['output']/'decisions_complete.json')
    n = len(data['sites']); selected = {s: {p: np.zeros(n, bool) for p in POLICIES} for s in cfg['seeds']}
    proposals = {s: np.empty((n, 12, 2)) for s in cfg['seeds']}
    fits, checked_choices, fitting_rows = [], 0, 0
    for key, meta in views.items():
        ids, x, y, d, pr, bits, w, q = training_arrays(meta, data, refs, key, old, 'ramp')
        cp, oldcp = states[key], neural[key, 'ramp']
        assert cp['step'] == 12000 and cp['settings'] == oldcp['settings'] == cfg['training']
        np.testing.assert_array_equal(cp['draws'], oldcp['draws'])
        np.testing.assert_array_equal(cp['loss_weights'], oldcp['loss_weights'])
        assert cp['draws'].sum() == 3072000 and not cp['draws'][~pr['known']].any()
        for field in ('mean', 'std', 'known', 'weights', 'constant'):
            np.testing.assert_array_equal(cp['preprocess'][field], oldcp['preprocess'][field])
            np.testing.assert_array_equal(cp['preprocess'][field], pr[field])
        assert cp['identity']['inputs_sha256'] == array_hash(ids, x, d, q)
        assert cp['identity']['labels_sha256'] == array_hash(y)
        with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); raw = z['prediction'].copy()
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2); alternative = manual_candidates(b, raw)[0]['ramp']
        target, valid = read_arrays(data, ids, 'target'), read_arrays(data, ids, 'valid')
        cv, _ = distances(b, target, valid, data['scale'][ids]); nc, _ = distances(alternative, target, valid, data['scale'][ids])
        native = np.column_stack((np.maximum(cv-nc, 0), np.maximum(nc-cv, 0))); native[~valid.all(1)] = np.nan
        np.testing.assert_allclose(native, y, rtol=1e-10, atol=1e-8, equal_nan=True)
        fraction_target = np.zeros_like(y); good = pr['known'] & (d > 0)
        fraction_target[good] = native[good]/d[good, None]
        fraction_target /= np.maximum(1., fraction_target.sum(1))[:, None]
        mass = cp['draws'].astype(float)*w*d/pr['cost_scale']; mass /= mass[mass > 0].mean()
        np.testing.assert_array_equal(mass, cp['expected_empirical_weight'])
        newrisk = empirical_square(predict_fractions(cp, x), fraction_target, mass)
        oldrisk = empirical_square(predict_fractions(oldcp, x), fraction_target, mass)
        ft = next(r for r in forest['training'] if r['view'] == key and r['arm'] == 'ramp')
        fits.append(dict(view=key, fitting_rows=len(ids), complete=int(pr['known'].sum()),
            empirical_fraction_square_new=newrisk, empirical_fraction_square_old_log=oldrisk,
            forest_fitting_fraction_square_cached_verified=ft['fit']['trace'][-1]['fitting_fraction_MSE']))
        fitting_rows += len(ids)
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            hi, raw = z['ids'].copy(), z['prediction'].copy()
        hb = data['geometry'][hi, 332:356].reshape(-1, 12, 2); proposal = manual_candidates(hb, raw)[0]['ramp']
        proposals[meta['seed']][hi] = proposal
        archive = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT/archive['path']) == archive['sha256']
        oldarchive = next(r for r in ranking['archives'] if r['view'] == key)
        with np.load(ROOT/oldarchive['path'], allow_pickle=False) as z: older = {k: z[k].copy() for k in z.files}
        with np.load(ROOT/archive['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(hi, z['ids'])
            score, dd = z['square_score'].copy(), z['distance'].copy()
            # Convex versus delta-form candidates differ at double roundoff; forward replay uses the frozen delta form.
            canonical = hb.astype(float)+(np.arange(12)/11)[None, :, None]*(raw.astype(float)-hb.astype(float))
            np.testing.assert_allclose(canonical, proposal, rtol=1e-12, atol=1e-12)
            hx, canonical_d, _ = features(data['geometry'][hi], canonical, data['scale'][hi])
            np.testing.assert_array_equal(dd, canonical_d)
            independent_d = np.sqrt(np.square(proposal-hb).sum(-1)).mean(1)*data['scale'][hi]
            np.testing.assert_allclose(dd, independent_d, rtol=1e-10, atol=1e-8)
            np.testing.assert_array_equal(native_scores(cp, hx, dd), score)
            history = data['geometry'][hi, :16].reshape(-1, 8, 2)
            eligible = (dd > 0) & np.any(history[:, -1] != history[:, -2], axis=1)
            expected = manual_choices(score, eligible, hi, int(older['forest_ratio'].sum()))
            for name in POLICIES:
                use = expected[name] if name.startswith('square') else older[name.replace('log_', 'neural_')]
                np.testing.assert_array_equal(use, z[name]); selected[meta['seed']][name][hi] = use; checked_choices += 1
        print(json.dumps(dict(view=key, state='targets_objective_exposure_choices_checked', new_fit_risk=newrisk, old_log_fit_risk=oldrisk)), flush=True)
    target, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    b = data['geometry'][:, 332:356].reshape(n, 12, 2); cv, cf = distances(b, target, valid, data['scale']); full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    for key, meta in views.items():
        use = data['sites'] == meta['outer_site']; pr = states[key]['preprocess']
        masks['hard'][use] = cv[use] >= pr['hard_cut']; masks['positive_easy'][use] = (cv[use] > 0) & (cv[use] <= pr['positive_easy_cut'])
    safe = np.where(valid[..., None], target, 0).astype(float)
    be = np.sqrt(np.square(b-safe).sum(-1))*data['scale'][:, None]; errors, bounds = {}, {}
    for seed, p in proposals.items():
        errors[seed] = distances(p, target, valid, data['scale'])
        pe = np.sqrt(np.square(p-safe).sum(-1))*data['scale'][:, None]
        dd = np.sqrt(np.square(p-b).sum(-1))*data['scale'][:, None]
        observed = np.where(valid, be-pe, 0).sum(1)/12; unknown = np.where(valid, 0, dd).sum(1)/12
        bounds[seed] = observed-unknown, observed+unknown
    reductions = 0
    for name in POLICIES:
        ades, fdes = [], []
        for seed in cfg['seeds']:
            use = selected[seed][name]; ade, fde = np.where(use, errors[seed][0], cv), np.where(use, errors[seed][1], cf)
            ades.append(ade); fdes.append(fde); r = report['summaries'][name]['seeds'][str(seed)]
            assert r['selected'] == int(use.sum()) and r['selected_unknown'] == int((use & ~valid.any(1)).sum())
            assert r['selected_incomplete'] == int((use & ~full).sum()) and r['zero_CV_harmed'] == int((ade[masks['zero_CV']] > 0).sum())
            reductions += check_reduction(ade, cv, data['sites'], cfg['sites'], r['ADE'])
            reductions += check_reduction(fde, cf, data['sites'], cfg['sites'], r['FDE'])
            for group, mask in masks.items():
                reductions += check_reduction(ade[mask], cv[mask], data['sites'][mask], cfg['sites'], r['subsets'][group])
            for site in cfg['sites']:
                np.testing.assert_allclose([np.where(use, v, 0)[data['sites']==site].mean() for v in bounds[seed]],
                    r['full_grid_absolute_gain_bounds'][site], rtol=1e-10, atol=1e-8)
        r = report['summaries'][name]
        reductions += check_reduction(np.mean(ades, 0), cv, data['sites'], cfg['sites'], r['ADE'])
        reductions += check_reduction(np.mean(fdes, 0), cf, data['sites'], cfg['sites'], r['FDE'])
        for group, mask in masks.items():
            reductions += check_reduction(np.mean(ades, 0)[mask], cv[mask], data['sites'][mask], cfg['sites'], r['subsets'][group])
    for key, contrast in report['contrasts'].items():
        a, c = key.split('_minus_')
        check_contrast(report['summaries'][a]['ADE']['by_scene'], report['summaries'][c]['ADE']['by_scene'], cfg['sites'], contrast)
    r = report['summaries']['square_strict']; contrast = report['contrasts']['square_strict_minus_log_strict']
    gates = dict(positive_primary_ci=contrast['ci95_pp'][0] > 0,
        each_seed_positive_cv=all(s['ADE']['equal_scene_gain_percent'] > 0 for s in r['seeds'].values()),
        aggregate_easy=all(s['subsets']['positive_easy']['equal_scene_gain_percent'] >= -2 for s in r['seeds'].values()),
        each_scene_seed_easy=all(v['gain_percent'] >= -2 for s in r['seeds'].values() for v in s['subsets']['positive_easy']['by_scene'].values()),
        exact_zero=not any(s['zero_CV_harmed'] for s in r['seeds'].values()))
    assert gates == report['primary_gates'] and all(gates.values()) == report['primary_joint_empirical_pass']
    assert_current(identity)
    immutable_json(public/'separate_verification.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
        verifier_sha256=file_digest(Path(__file__)), all_checks_passed=True, fits_checked=12,
        fitting_rows_checked=fitting_rows, choices_checked=checked_choices, scene_reductions_checked=reductions,
        held_score_rows_separately_reconstructed=n*len(cfg['seeds']),
        held_score_forward_exact_shared_float32_kernels=True,
        empirical_fit_objective=fits, score_checkpoints_replayed_by_runner=True,
        context_smoothness_conditional_quality_replayed_only=True, independent_research_confirmation=False))
    print(json.dumps(dict(verified=True, fits=12, decisions=checked_choices, reductions=reductions)))


if __name__ == '__main__': main()
