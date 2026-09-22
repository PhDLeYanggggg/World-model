"""Fixed forest/neural risk comparison, with choices frozen before label readout."""
from pathlib import Path
import numpy as np
from scripts.run_m3w_bounded_cost import features, read_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from scripts.run_m3w_native_nested import write_arrays
from src.world_model.m3w_forest_cost_head import predict, selections, ARMS, POLICIES
from src.world_model.m3w_temporal_intervention import candidates, second_difference
from src.evaluation.m3w_fixed_choice_context import audit as context_audit
from src.evaluation.m3w_conditional_cost_eval import gate
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds
from src.evaluation.m3w_temporal_fit_support import population_summary

ROOT = Path(__file__).resolve().parents[2]


def evaluate(cfg, data, views, predictions, previous, identity, records, states, beat, verify=False):
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']; n = len(data['sites'])
    choices = {s: {p: np.zeros(n, bool) for p in POLICIES} for s in cfg['seeds']}
    proposals = {s: {a: np.empty((n, 12, 2), float) for a in ARMS} for s in cfg['seeds']}
    distances = {s: {} for s in cfg['seeds']}
    archives, scores = [], {}
    old_archives = {r['view']: r for r in previous['archives']}
    for key, meta in views.items():
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            ids, p = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        transformed, _ = candidates(b, p)
        with np.load(ROOT/old_archives[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); old = {k: z[k].copy() for k in z.files}
        arrays = dict(ids=ids)
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        for arm in ARMS:
            x, d, _ = features(data['geometry'][ids], transformed[arm], data['scale'][ids])
            cp = states[key, arm]; score = predict(cp['model'], x, d, cp['preprocess'])
            neural = old[arm+'_score']; bits = selections(score, neural, past, d, ids)
            np.testing.assert_array_equal(bits['neural'], old[arm+'_strict'])
            proposals[meta['seed']][arm][ids] = transformed[arm]
            distances[meta['seed']].setdefault(arm, np.empty(n))[ids] = d
            arrays[arm+'_forest_score'], arrays[arm+'_neural_score'] = score, neural
            arrays[arm+'_distance'] = d; scores[key, arm] = (score, neural)
            for name, use in bits.items():
                name = arm+'_'+name; arrays[name] = use; choices[meta['seed']][name][ids] = use
        path = root/'decisions'/(key+'.npz')
        if verify and not path.exists(): raise ValueError('Cannot replay missing decisions')
        write_arrays(path, arrays)
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
        beat(state='decisions_replayed' if verify else 'decisions_frozen', view=key)
    immutable_json(root/'decisions_complete.json', dict(identity=identity, archives=archives,
        future_targets_used_in_decisions=False, future_mask_used_in_decisions=False))
    interaction = context_audit(cfg, data, proposals, choices, {p: p.split('_')[0] for p in POLICIES}, beat)
    y, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    baseline = data['geometry'][:, 332:356].reshape(n, 12, 2)
    past = data['geometry'][:, :16].reshape(n, 8, 2)
    cv, cf = native_errors(baseline, y, valid, data['scale']); full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    for key, meta in views.items():
        ix = data['sites'] == meta['outer_site']; pr = states[key, 'ramp']['preprocess']
        masks['hard'][ix] = cv[ix] >= pr['hard_cut']
        masks['positive_easy'][ix] = (cv[ix] > 0) & (cv[ix] <= pr['positive_easy_cut'])
    errors, bounds, smooth = {}, {}, {}
    cv_smooth = second_difference(past, baseline, data['scale'])
    for seed in cfg['seeds']:
        for arm in ARMS:
            p = proposals[seed][arm]
            errors[seed, arm] = native_errors(p, y, valid, data['scale'])
            bounds[seed, arm] = partial_gain_bounds(p, baseline, y, valid, data['scale'])
            smooth[seed, arm] = second_difference(past, p, data['scale'])
    def metric(a, r, mask=None):
        if mask is None: mask = np.ones(n, bool)
        return paired_scene_metrics(a[mask], r[mask], data['sites'][mask], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=cfg['bootstrap_resamples'])
    summaries = {}
    for name in POLICIES:
        arm = name.split('_')[0]; seeds, ades, fdes = {}, [], []
        for seed in cfg['seeds']:
            use = choices[seed][name]
            ade, fde = np.where(use, errors[seed, arm][0], cv), np.where(use, errors[seed, arm][1], cf)
            sm = np.where(use, smooth[seed, arm], cv_smooth); ades.append(ade); fdes.append(fde)
            seeds[str(seed)] = dict(ADE=metric(ade, cv), FDE=metric(fde, cf), selected=int(use.sum()),
                selected_unknown=int((use & ~valid.any(1)).sum()), selected_incomplete=int((use & ~full).sum()),
                zero_CV_harmed=int((ade[masks['zero_CV']] > 0).sum()),
                full_grid_absolute_gain_bounds={s: [float(np.where(use, bounds[seed, arm][k], 0)[data['sites']==s].mean())
                    for k in ('lower', 'upper')] for s in cfg['sites']},
                causal_displacement_sum={s: float(np.where(use, distances[seed][arm], 0)[data['sites']==s].sum()) for s in cfg['sites']},
                smoothness={s: dict(mean=float(sm[data['sites']==s].mean()), p95=float(np.quantile(sm[data['sites']==s], .95)),
                    cv_mean=float(cv_smooth[data['sites']==s].mean())) for s in cfg['sites']},
                subsets={g: metric(ade, cv, m) for g, m in masks.items()})
        summaries[name] = dict(ADE=metric(np.mean(ades, 0), cv), FDE=metric(np.mean(fdes, 0), cf), seeds=seeds,
            subsets={g: metric(np.mean(ades, 0), cv, m) for g, m in masks.items()})
        # This is an exact cached-result check, not a new independently trained neural comparator.
        if name.endswith('_neural'):
            original = previous['summaries'][arm+'_strict']
            for field in ('ADE', 'FDE', 'subsets'): assert summaries[name][field] == original[field]
        beat(state='policy_metrics_complete', policy=name)
    def contrast(a, b):
        return paired_scene_contrast([a['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']],
                                     [b['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']])
    contrasts = {a: {c: contrast(summaries[a+'_forest'], summaries[a+'_'+c])
                    for c in ('neural', 'neural_matched')} for a in ARMS}
    quality = []
    for key, meta in views.items():
        ids = np.flatnonzero(data['sites'] == meta['outer_site'])
        for arm in ARMS:
            delta = cv[ids] - errors[meta['seed'], arm][0][ids]
            target = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0))); target[~full[ids]] = np.nan
            for estimator, score in zip(('forest', 'neural'), scores[key, arm]):
                for chosen_by in ('forest', 'neural'):
                    use = choices[meta['seed']][arm+'_'+chosen_by][ids]
                    quality.append(dict(view=key, arm=arm, estimator=estimator, chosen_by=chosen_by,
                        costs=population_summary(target, score, distances[meta['seed']][arm][ids], use, np.ones(len(ids)))))
    checks = gate(summaries['ramp_forest'], contrasts['ramp']['neural'])
    result = dict(identity=identity, result_source='fresh_24_forests_fixed_source_readout',
        reference_source='cached_verified_temporal_neural_heads_and_nested_forecasts',
        summaries=summaries, contrasts=contrasts, prior_scalar_reference=previous['scalar_reference'],
        prior_uncontrolled={a: previous['summaries'][a+'_uncontrolled'] for a in ARMS},
        interaction=interaction, archives=archives, conditional_quality=quality,
        training=[dict(view=k, arm=a, **{f: r[f] for f in ('fit', 'checkpoint', 'checkpoint_sha256')})
                  for (k, a), r in records.items()],
        primary_gates=checks, primary_joint_empirical_pass=all(checks.values()),
        decision_manifest_sha256=file_digest(root/'decisions_complete.json'),
        independent_confirmation=False, risk_calibrated=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity); immutable_json(public/'analysis.json', result)
    if verify:
        immutable_json(public/'replay.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
            checkpoint_endpoints_replayed=24, score_rows=n*len(cfg['seeds'])*2, all_checks_passed=True))
    beat(state='verified' if verify else 'evaluated', primary_gates=checks)
