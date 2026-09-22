"""Fixed matched temporal intervention evaluation; no post-readout policy choice."""
from pathlib import Path
import numpy as np
from scripts.run_m3w_bounded_cost import features, read_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from scripts.run_m3w_native_nested import write_arrays
from src.world_model.m3w_bounded_cost_head import build, predict
from src.world_model.m3w_temporal_intervention import candidates, selections, ARMS, POLICIES, policy_arm, second_difference
from src.evaluation.m3w_temporal_interaction_audit import audit as interaction_audit
from src.evaluation.m3w_conditional_cost_eval import gate
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds

ROOT = Path(__file__).resolve().parents[2]


def prefix_errors(p, y, valid, scale):
    e = np.linalg.norm(p-np.where(valid[..., None], y, 0), axis=-1)*scale[:, None]
    count = np.cumsum(valid, axis=1)
    return np.divide(np.cumsum(np.where(valid, e, 0), axis=1), count,
                     out=np.full(count.shape, np.nan), where=count>0)


def evaluate(cfg, data, views, predictions, scalar, identity, records, states, beat, verify=False):
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']; n = len(data['sites'])
    choices = {s: {p: np.zeros(n, bool) for p in POLICIES} for s in cfg['seeds']}
    proposals = {s: {a: np.empty((n, 12, 2), float) for a in ARMS} for s in cfg['seeds']}
    distances = {s: np.empty(n, float) for s in cfg['seeds']}; archives = []
    for key, meta in views.items():
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            ids, p = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        transformed, alpha = candidates(b, p); scores, d = {}, {}
        for arm in ARMS:
            x, d[arm], _ = features(data['geometry'][ids], transformed[arm], data['scale'][ids])
            cp = states[key, arm]; model = build(x.shape[1], cfg['training']['width'], meta['seed'])
            model.load_state_dict(cp['model']); scores[arm] = predict(model, x, d[arm], cp['preprocess'], 'bounded_native')
            proposals[meta['seed']][arm][ids] = transformed[arm]
        np.testing.assert_allclose(d['ramp'], d['uniform'], rtol=1e-10, atol=1e-8)
        np.testing.assert_array_equal(d['ramp'] == 0, d['uniform'] == 0)
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        bits = selections(scores['ramp'], scores['uniform'], past, d['ramp'], ids)
        path = root/'decisions'/(key+'.npz')
        if verify and not path.exists():
            raise ValueError('Cannot replay missing decisions')
        write_arrays(path, dict(ids=ids, distance=d['ramp'], uniform_alpha=alpha,
                               ramp_score=scores['ramp'], uniform_score=scores['uniform'], **bits))
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
        for name, bits_ in bits.items():
            choices[meta['seed']][name][ids] = bits_
        distances[meta['seed']][ids] = d['ramp']
        beat(state='replayed' if verify else 'decisions_frozen', view=key)
    immutable_json(root/'decisions_complete.json', dict(identity=identity, archives=archives,
                   future_targets_used_in_decisions=False, future_mask_used_in_decisions=False))
    interaction = interaction_audit(cfg, data, proposals, choices, beat)
    y, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    baseline = data['geometry'][:, 332:356].reshape(n, 12, 2)
    past = data['geometry'][:, :16].reshape(n, 8, 2)
    cv, cf = native_errors(baseline, y, valid, data['scale']); full = valid.all(1)
    cv_prefix = prefix_errors(baseline.astype(float), y, valid, data['scale'])
    cv_smooth = second_difference(past, baseline, data['scale'])
    masks = dict(complete=full, zero_CV=full & (cv==0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    for key, meta in views.items():
        ix = data['sites'] == meta['outer_site']; pr = states[key, 'ramp']['preprocess']
        masks['hard'][ix] = cv[ix]>=pr['hard_cut']; masks['positive_easy'][ix] = (cv[ix]>0) & (cv[ix]<=pr['positive_easy_cut'])
    errors, bounds, prefixes, smooth = {}, {}, {}, {}
    for seed in cfg['seeds']:
        for arm in ARMS:
            p = proposals[seed][arm]
            errors[seed, arm] = native_errors(p, y, valid, data['scale'])
            bounds[seed, arm] = partial_gain_bounds(p, baseline, y, valid, data['scale'])
            prefixes[seed, arm] = prefix_errors(p, y, valid, data['scale'])
            smooth[seed, arm] = second_difference(past, p, data['scale'])
    def metric(a, ref, mask=None):
        if mask is None:
            mask = np.ones(n, bool)
        return paired_scene_metrics(a[mask], ref[mask], data['sites'][mask], expected_scenes=cfg['sites'],
                                    dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=3000)
    summaries = {}
    for name in POLICIES:
        arm = policy_arm(name); seeds, ades, fdes, pref = {}, [], [], []
        for seed in cfg['seeds']:
            use = choices[seed][name]
            ade, fde = np.where(use, errors[seed, arm][0], cv), np.where(use, errors[seed, arm][1], cf)
            pp = np.where(use[:, None], prefixes[seed, arm], cv_prefix)
            sm = np.where(use, smooth[seed, arm], cv_smooth)
            ades.append(ade); fdes.append(fde); pref.append(pp)
            seeds[str(seed)] = dict(ADE=metric(ade, cv), FDE=metric(fde, cf), selected=int(use.sum()),
                selected_unknown=int((use & ~valid.any(1)).sum()), selected_incomplete=int((use & ~full).sum()),
                zero_CV_harmed=int((ade[masks['zero_CV']]>0).sum()),
                full_grid_absolute_gain_bounds={site:[float(np.where(use, bounds[seed, arm][k], 0)[data['sites']==site].mean())
                    for k in ('lower', 'upper')] for site in cfg['sites']},
                causal_displacement_sum={s:float(np.where(use, distances[seed], 0)[data['sites']==s].sum()) for s in cfg['sites']},
                smoothness={s:dict(mean=float(sm[data['sites']==s].mean()), p95=float(np.quantile(sm[data['sites']==s], .95)),
                                   cv_mean=float(cv_smooth[data['sites']==s].mean())) for s in cfg['sites']},
                subsets={g:metric(ade, cv, m) for g, m in masks.items()})
        summaries[name] = dict(ADE=metric(np.mean(ades, 0), cv), FDE=metric(np.mean(fdes, 0), cf), seeds=seeds,
                              subsets={g:metric(np.mean(ades, 0), cv, m) for g, m in masks.items()},
                              prefix_ADE={str(k+1):metric(np.mean(pref, 0)[:, k], cv_prefix[:, k]) for k in range(12)})
        beat(state='policy_metrics_complete', policy=name)
    def contrast(a, b):
        return paired_scene_contrast([a['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']],
                                     [b['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']])
    contrasts = {p:contrast(summaries['ramp_strict'], summaries[p]) for p in POLICIES if p != 'ramp_strict'}
    contrasts['scalar_log_strict'] = contrast(summaries['ramp_strict'], scalar['summaries']['strict_stop'])
    checks = gate(summaries['ramp_strict'], contrasts['uniform_strict'])
    checks['positive_old_scalar_ci'] = contrasts['scalar_log_strict']['ci95_pp'][0]>0
    training = [dict(view=k, arm=a, **{f:r[f] for f in ('fit','rows','supported_rows','checkpoint','checkpoint_sha256')})
                for (k, a), r in records.items()]
    result = dict(identity=identity, result_source='fresh_24_cost_fits_fixed_temporal_readout',
                  reference_source='cached_verified_nested_forecasts_and_scalar_log_reference',
                  summaries=summaries, contrasts=contrasts, scalar_reference=scalar['summaries']['strict_stop'],
                  interaction=interaction, archives=archives, training=training, primary_gates=checks,
                  primary_joint_empirical_pass=all(checks.values()), new_updates=sum(r['fit']['step'] for r in records.values()),
                  new_draws=sum(r['fit']['total_draws'] for r in records.values()),
                  decision_manifest_sha256=file_digest(root/'decisions_complete.json'),
                  independent_confirmation=False, risk_calibrated=False, deployment=False,
                  stage5c_executed=False, smc_enabled=False)
    assert result['new_updates']==288000 and result['new_draws']==73728000
    assert_current(identity); immutable_json(public/'analysis.json', result)
    if verify:
        immutable_json(public/'replay.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
                       checkpoint_endpoints_replayed=24, score_rows=n*len(cfg['seeds'])*2,
                       all_checks_passed=True))
    beat(state='verified' if verify else 'evaluated', primary_gates=checks)
