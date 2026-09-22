"""Fixed final-budget neural objective comparison; no held model/threshold selection."""
from pathlib import Path
import numpy as np
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from scripts.run_m3w_native_nested import write_arrays
from scripts.run_m3w_bounded_cost import features, read_arrays
from src.world_model.m3w_bounded_cost_head import build, predict
from src.world_model.m3w_temporal_intervention import candidates, second_difference
from src.evaluation.m3w_risk_ranking import fixed_count
from src.evaluation.m3w_conditional_cost_audit import strict_bits
from src.evaluation.m3w_conditional_cost_eval import gate
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_fixed_choice_context import audit as context_audit
from src.evaluation.m3w_temporal_fit_support import population_summary
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds
from src.evaluation.m3w_experiment_contract import file_digest

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ('square_strict', 'square_ratio', 'square_gain', 'log_strict', 'log_ratio', 'log_gain', 'forest_ratio', 'forest_gain')


def evaluate(cfg, data, views, predictions, states, records, forest, ranking, identity, beat, verify=False):
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']; n = len(data['sites'])
    chosen = {s: {p: np.zeros(n, bool) for p in POLICIES} for s in cfg['seeds']}
    proposals = {s: {'ramp': np.empty((n, 12, 2))} for s in cfg['seeds']}
    distance = {s: np.empty(n) for s in cfg['seeds']}; archives, scores = [], {}
    for key, meta in views.items():
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            ids, raw = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2); p = candidates(b, raw)[0]['ramp']
        x, d, _ = features(data['geometry'][ids], p, data['scale'][ids]); cp = states[key]
        model = build(x.shape[1], cfg['training']['width'], meta['seed']); model.load_state_dict(cp['model'])
        score = predict(model, x, d, cp['preprocess'], 'bounded_native')
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        pool = (d > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
        archive = next(r for r in ranking['archives'] if r['view'] == key)
        with np.load(ROOT/archive['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); old = {k: z[k].copy() for k in z.files}
        farch = next(r for r in forest['archives'] if r['view'] == key)
        with np.load(ROOT/farch['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); fscore, nnscore = z['ramp_forest_score'].copy(), z['ramp_neural_score'].copy()
        k = int(old['forest_ratio'].sum())
        bits = dict(square_strict=strict_bits(score, past, d), square_ratio=fixed_count(score, pool, ids, k, 'ratio'),
            square_gain=fixed_count(score, pool, ids, k, 'gain'), log_strict=old['neural_strict'],
            log_ratio=old['neural_ratio'], log_gain=old['neural_gain'], forest_ratio=old['forest_ratio'], forest_gain=old['forest_gain'])
        path = root/'decisions'/(key+'.npz')
        if verify and not path.exists(): raise ValueError('Cannot replay absent decisions')
        write_arrays(path, dict(ids=ids, square_score=score, distance=d, **bits))
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
        for name, use in bits.items(): chosen[meta['seed']][name][ids] = use
        proposals[meta['seed']]['ramp'][ids] = p; distance[meta['seed']][ids] = d
        scores[key] = dict(square=score, log=nnscore, forest=fscore)
        beat(state='decisions_replayed' if verify else 'decisions_frozen', view=key)
    immutable_json(root/'decisions_complete.json', dict(identity=identity, archives=archives,
        future_targets_used_in_choices=False, future_masks_used_in_choices=False))
    interaction = context_audit(cfg, data, proposals, chosen, {p: 'ramp' for p in POLICIES}, beat)
    y, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    b = data['geometry'][:, 332:356].reshape(n, 12, 2); past = data['geometry'][:, :16].reshape(n, 8, 2)
    cv, cf = native_errors(b, y, valid, data['scale']); full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    for key, meta in views.items():
        ix = data['sites'] == meta['outer_site']; pr = states[key]['preprocess']
        masks['hard'][ix] = cv[ix] >= pr['hard_cut']
        masks['positive_easy'][ix] = (cv[ix] > 0) & (cv[ix] <= pr['positive_easy_cut'])
    errors, bounds, smooth = {}, {}, {}
    cv_smooth = second_difference(past, b, data['scale'])
    for seed in cfg['seeds']:
        p = proposals[seed]['ramp']; errors[seed] = native_errors(p, y, valid, data['scale'])
        bounds[seed] = partial_gain_bounds(p, b, y, valid, data['scale']); smooth[seed] = second_difference(past, p, data['scale'])
    def metric(a, r, mask=None):
        if mask is None: mask = np.ones(n, bool)
        return paired_scene_metrics(a[mask], r[mask], data['sites'][mask], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=cfg['bootstrap_resamples'])
    summaries = {}
    for name in POLICIES:
        seeds, ades, fdes = {}, [], []
        for seed in cfg['seeds']:
            use = chosen[seed][name]; ade = np.where(use, errors[seed][0], cv); fde = np.where(use, errors[seed][1], cf)
            sm = np.where(use, smooth[seed], cv_smooth); ades.append(ade); fdes.append(fde)
            seeds[str(seed)] = dict(ADE=metric(ade, cv), FDE=metric(fde, cf), selected=int(use.sum()),
                selected_unknown=int((use & ~valid.any(1)).sum()), selected_incomplete=int((use & ~full).sum()),
                zero_CV_harmed=int((ade[masks['zero_CV']] > 0).sum()),
                full_grid_absolute_gain_bounds={s: [float(np.where(use, bounds[seed][k], 0)[data['sites']==s].mean())
                    for k in ('lower', 'upper')] for s in cfg['sites']},
                causal_displacement_sum={s: float(np.where(use, distance[seed], 0)[data['sites']==s].sum()) for s in cfg['sites']},
                smoothness={s: dict(mean=float(sm[data['sites']==s].mean()), p95=float(np.quantile(sm[data['sites']==s], .95)),
                    cv_mean=float(cv_smooth[data['sites']==s].mean())) for s in cfg['sites']},
                subsets={g: metric(ade, cv, m) for g, m in masks.items()})
        summaries[name] = dict(ADE=metric(np.mean(ades, 0), cv), FDE=metric(np.mean(fdes, 0), cf), seeds=seeds,
            subsets={g: metric(np.mean(ades, 0), cv, m) for g, m in masks.items()})
        if not name.startswith('square'):
            oldname = name.replace('log_', 'neural_')
            assert summaries[name] == ranking['summaries'][oldname]
        beat(state='metrics_complete', policy=name)
    def contrast(a, c):
        return paired_scene_contrast([summaries[a]['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']],
            [summaries[c]['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']])
    pairs = (('square_strict', 'log_strict'), ('square_ratio', 'log_ratio'),
             ('square_gain', 'log_gain'), ('square_ratio', 'forest_ratio'), ('square_gain', 'forest_gain'))
    contrasts = {a+'_minus_'+c: contrast(a, c) for a, c in pairs}; quality = []
    for key, meta in views.items():
        ids = np.flatnonzero(data['sites'] == meta['outer_site']); seed = meta['seed']
        delta = cv[ids]-errors[seed][0][ids]
        target = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0))); target[~full[ids]] = np.nan
        for estimator, score in scores[key].items():
            for choice in ('square_strict', 'square_ratio', 'log_strict', 'log_ratio', 'forest_ratio'):
                quality.append(dict(view=key, estimator=estimator, chosen_by=choice,
                    costs=population_summary(target, score, distance[seed][ids], chosen[seed][choice][ids], np.ones(len(ids)))))
    gates = gate(summaries['square_strict'], contrasts['square_strict_minus_log_strict'])
    result = dict(identity=identity, result_source='fresh_12_neural_fraction_square_fits_fixed_readout',
        reference_source='cached_verified_forest_and_log_heads_nested_forecasts',
        summaries=summaries, contrasts=contrasts, conditional_quality=quality, interaction=interaction, archives=archives,
        training=[dict(view=k, **{f: r[f] for f in ('fit', 'checkpoint', 'checkpoint_sha256')}) for k, r in records.items()],
        primary_gates=gates, primary_joint_empirical_pass=all(gates.values()),
        decision_manifest_sha256=file_digest(root/'decisions_complete.json'),
        original_forest_primary_gate_unchanged=forest['primary_gates'], independent_confirmation=False,
        risk_calibrated=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity); immutable_json(public/'analysis.json', result)
    if verify:
        immutable_json(public/'replay.json', dict(analysis_sha256=file_digest(public/'analysis.json'), all_checks_passed=True,
            checkpoints_replayed=12, score_rows=n*len(cfg['seeds']), choices_replayed=len(views)*len(POLICIES)))
    beat(state='verified' if verify else 'evaluated', primary_gates=gates)
