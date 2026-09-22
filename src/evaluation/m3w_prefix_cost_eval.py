"""Frozen prefix-profile readout; prediction decisions precede outcome access."""
from pathlib import Path
import numpy as np

from scripts.run_m3w_bounded_cost import features, read_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from scripts.run_m3w_native_nested import write_arrays
from src.world_model.m3w_prefix_cost_head import build, predict, ARMS
from src.world_model.m3w_prefix_cost_targets import causal_prefix_disagreement, supervised_prefix_costs
from src.evaluation.m3w_prefix_cost_policy import selections, POLICIES
from src.evaluation.m3w_conditional_cost_eval import gate
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds

ROOT = Path(__file__).resolve().parents[2]


def evaluate(cfg, data, views, predictions, scalar, refs, identity, records, states, beat, verify=False):
    root, public = ROOT / cfg['output'], ROOT / cfg['reports']
    n = len(data['sites'])
    chosen = {s: {p: np.zeros(n, bool) for p in POLICIES} for s in cfg['seeds']}
    candidates = {s: np.empty((n, 12, 2), np.float32) for s in cfg['seeds']}
    archives, profiles = [], {}
    for key, meta in views.items():
        with np.load(ROOT / predictions[key]['path'], allow_pickle=False) as z:
            ids, p = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        d = causal_prefix_disagreement(b, p, data['scale'][ids])
        x, terminal_d, _ = features(data['geometry'][ids], p, data['scale'][ids])
        np.testing.assert_allclose(d[:, -1], terminal_d, rtol=1e-10, atol=1e-8)
        scores = {}
        for arm in ARMS:
            cp = states[key, arm]
            model = build(x.shape[1], cfg['training']['width'], meta['seed'])
            model.load_state_dict(cp['model'])
            distances = np.repeat(d[:, -1:], 12, 1) if arm == 'terminal_repeat' else d
            scores[arm] = predict(model, x, distances, cp['preprocess'])
            assert np.all(scores[arm].sum(-1) <= distances * (1+2e-6) + 1e-8)
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        bits = selections(scores['terminal_repeat'], scores['prefix'], past, d, ids)
        path = root / 'decisions' / (key + '.npz')
        if verify and not path.exists():
            raise ValueError('Cannot replay missing decisions')
        write_arrays(path, dict(ids=ids, distance=d, control=scores['terminal_repeat'], profile=scores['prefix'], **bits))
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
        for name, use in bits.items():
            chosen[meta['seed']][name][ids] = use
        candidates[meta['seed']][ids] = p
        profiles[key] = scores
        beat(state='replayed' if verify else 'decisions_frozen', view=key)
    immutable_json(root / 'decisions_complete.json', dict(identity=identity, archives=archives,
                   future_targets_used_in_decisions=False, future_mask_used_in_decisions=False))

    y, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    baseline = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = native_errors(baseline, y, valid, data['scale'])
    full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    for site in cfg['sites']:
        prs = [states[k, 'prefix']['preprocess'] for k, m in views.items() if m['outer_site'] == site]
        for pr in prs[1:]:
            assert pr['hard_cut'] == prs[0]['hard_cut'] and pr['positive_easy_cut'] == prs[0]['positive_easy_cut']
        ix = data['sites'] == site
        masks['hard'][ix] = cv[ix] >= prs[0]['hard_cut']
        masks['positive_easy'][ix] = (cv[ix] > 0) & (cv[ix] <= prs[0]['positive_easy_cut'])
    errors = {s: native_errors(p, y, valid, data['scale']) for s, p in candidates.items()}
    bounds = {s: partial_gain_bounds(p, baseline, y, valid, data['scale']) for s, p in candidates.items()}
    def metric(a, ref, mask=None):
        if mask is None:
            mask = np.ones(n, bool)
        return paired_scene_metrics(a[mask], ref[mask], data['sites'][mask], expected_scenes=cfg['sites'],
                                    dataset='sdd', coordinate_unit='annotation_pixel',
                                    bootstrap_resamples=cfg['bootstrap_resamples'])
    summaries = {}
    for policy in POLICIES:
        seeds, ades, fdes = {}, [], []
        for seed in cfg['seeds']:
            use = chosen[seed][policy]
            ade, fde = np.where(use, errors[seed][0], cv), np.where(use, errors[seed][1], cf)
            ades.append(ade); fdes.append(fde)
            seeds[str(seed)] = dict(ADE=metric(ade, cv), FDE=metric(fde, cf), selected=int(use.sum()),
                selected_unknown=int((use & ~valid.any(1)).sum()), selected_incomplete=int((use & ~full).sum()),
                zero_CV_harmed=int((ade[masks['zero_CV']] > 0).sum()),
                full_grid_absolute_gain_bounds={site: [float(np.where(use, bounds[seed][k], 0)[data['sites'] == site].mean())
                    for k in ('lower', 'upper')] for site in cfg['sites']},
                subsets={g: metric(ade, cv, m) for g, m in masks.items()})
        summaries[policy] = dict(ADE=metric(np.mean(ades, 0), cv), FDE=metric(np.mean(fdes, 0), cf), seeds=seeds,
                                subsets={g: metric(np.mean(ades, 0), cv, m) for g, m in masks.items()})
    def contrast(a, b):
        return paired_scene_contrast([a['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']],
                                     [b['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']])
    contrasts = {p: contrast(summaries['profile_guard'], summaries[p]) for p in POLICIES if p != 'profile_guard'}
    contrasts['scalar_log_strict'] = contrast(summaries['profile_guard'], scalar['summaries']['strict_stop'])
    checks = gate(summaries['profile_guard'], contrasts['control_terminal'])
    checks['positive_equal_count_ci'] = contrasts['control_matched']['ci95_pp'][0] > 0
    quality = []
    for key, meta in views.items():
        ids = np.flatnonzero(data['sites'] == meta['outer_site'])
        labels = supervised_prefix_costs(baseline[ids], candidates[meta['seed']][ids], y[ids], valid[ids], data['scale'][ids])
        for policy in ('control_terminal', 'profile_terminal', 'profile_guard'):
            arm = 'terminal_repeat' if policy == 'control_terminal' else 'prefix'
            score, use = profiles[key][arm], chosen[meta['seed']][policy][ids]
            for k in range(12):
                # Complete rows isolate the profile from future-support selection bias.
                ok = use & full[ids]
                row = dict(view=key, policy=policy, prefix=k+1, complete_selected=int(ok.sum()))
                if ok.any():
                    target = labels['costs'][ok, k]
                    row.update(predicted_harm=float(score[ok, k, 1].mean()), realized_harm=float(target[:, 1].mean()),
                               predicted_benefit=float(score[ok, k, 0].mean()), realized_benefit=float(target[:, 0].mean()),
                               MSE=float(((score[ok, k] - target)**2).mean()))
                observed = use & labels['available'][:, k] & (valid[ids].sum(1) == k+1) & ~full[ids]
                row['selected_exact_partial_prefix'] = int(observed.sum())
                if observed.any():
                    row['partial_predicted_harm'] = float(score[observed, k, 1].mean())
                    row['partial_realized_harm'] = float(labels['costs'][observed, k, 1].mean())
                quality.append(row)
        beat(state='prefix_diagnostics_complete', view=key)
    training = [dict(view=k, arm=a, **{f: r[f] for f in ('fit', 'rows', 'supported_rows', 'checkpoint', 'checkpoint_sha256')})
                for (k, a), r in records.items()]
    result = dict(identity=identity, result_source='fresh_24_risk_head_fits_fixed_development_readout',
                  reference_source='cached_verified_nested_forecasts_scalar_log_heads',
                  summaries=summaries, contrasts=contrasts, scalar_reference=scalar['summaries']['strict_stop'],
                  archives=archives, training=training, prefix_quality=quality, primary_gates=checks,
                  primary_joint_empirical_pass=all(checks.values()),
                  new_updates=sum(r['fit']['step'] for r in records.values()),
                  new_draws=sum(r['fit']['total_draws'] for r in records.values()),
                  decision_manifest_sha256=file_digest(root / 'decisions_complete.json'),
                  independent_confirmation=False, risk_calibrated=False, deployment=False,
                  stage5c_executed=False, smc_enabled=False)
    assert result['new_updates'] == 288000 and result['new_draws'] == 73728000
    assert_current(identity); immutable_json(public / 'analysis.json', result)
    if verify:
        immutable_json(public / 'replay.json', dict(analysis_sha256=file_digest(public / 'analysis.json'),
                       checkpoint_endpoints_replayed=24, score_profiles=n*len(cfg['seeds'])*2,
                       all_checks_passed=True))
    beat(state='verified' if verify else 'evaluated', primary_gates=checks)
