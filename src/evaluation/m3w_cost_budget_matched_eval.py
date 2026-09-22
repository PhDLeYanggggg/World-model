"""Fixed three-objective readout at common capacity, draws and duration."""
import numpy as np
from scripts import run_m3w_bounded_cost as bounded
from scripts.run_m3w_tempered_cost import causal_scores
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from scripts.run_m3w_native_nested import write_arrays
from src.world_model.m3w_bounded_cost_head import build, predict
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds
from src.evaluation.m3w_cost_support_audit import fit_cuts, strata, cost_stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('native', 'fraction', 'tempered')
POLICIES = ('net_stop', 'strict_stop', 'matched_count')


def contribution_gate(summary, contrasts):
    return dict(exact_zero=all(r['zero_CV_harmed'] == 0 for r in summary['seeds'].values()),
        easy=all(-r['subsets']['positive_easy']['equal_scene_gain_percent'] <= 2 for r in summary['seeds'].values()),
        each_seed_positive_cv=all(r['ADE']['equal_scene_gain_percent'] > 0 for r in summary['seeds'].values()),
        positive_vs_native_ci=contrasts['tempered_minus_native']['ci95_pp'][0] > 0,
        positive_vs_fraction_ci=contrasts['tempered_minus_fraction']['ci95_pp'][0] > 0)


def evaluate(cfg, data, views, predictions, controls, prior, identity, records, states, beat, verify=False):
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']
    n = len(data['sites'])
    chosen = {s:{a:{p:np.zeros(n, bool) for p in POLICIES} for a in ARMS} for s in cfg['seeds']}
    candidates = {s:np.empty((n, 12, 2), np.float32) for s in cfg['seeds']}
    old = {r['view']:r for r in prior['archives']}
    refs = {r['view']:r for r in controls['archives']}
    archives, scores, distances = [], {}, {}
    for key, meta in views.items():
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            ids, prediction = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        with np.load(ROOT/refs[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            anchor = z[cfg['matched_count_reference']].copy()
        arrays = dict(ids=ids, anchor=anchor)
        for arm in ARMS:
            cp = states[key, arm]
            score, distance = causal_scores(data, ids, prediction, cp, cfg)
            past = data['geometry'][ids, :16].reshape(-1, 8, 2)
            choices = bounded.selections(score, past, distance, anchor, ids)
            arrays[arm+'_score'] = score
            for policy, bits in choices.items():
                arrays[arm+'_'+policy] = bits
                chosen[meta['seed']][arm][policy][ids] = bits
            scores[key, arm] = score
            if arm == 'tempered':
                with np.load(ROOT/old[key]['path'], allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['wide_long_score'], score)
                    for policy in POLICIES:
                        np.testing.assert_array_equal(z['wide_long_'+policy], choices[policy])
        arrays['distance'] = distance
        distances[key] = distance
        candidates[meta['seed']][ids] = prediction
        path = root/'decisions'/f'{key}.npz'
        if verify and not path.exists():
            raise ValueError('Cannot replay a missing decision archive')
        write_arrays(path, arrays)
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
        beat(state='replayed' if verify else 'decisions_frozen', view=key)
    immutable_json(root/'decisions_complete.json', dict(identity=identity, archives=archives,
        future_targets_used_in_decisions=False, score_rows=n*len(cfg['seeds'])*3))
    y, valid = bounded.read_arrays(data, np.arange(n), 'target'), bounded.read_arrays(data, np.arange(n), 'valid')
    baseline = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = native_errors(baseline, y, valid, data['scale'])
    full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    errors = {s:native_errors(p, y, valid, data['scale']) for s,p in candidates.items()}
    bounds = {s:partial_gain_bounds(p, baseline, y, valid, data['scale']) for s,p in candidates.items()}
    diagnostics, quality = [], []
    for key, meta in views.items():
        ids = np.flatnonzero(data['sites'] == meta['outer_site'])
        pr = states[key, 'tempered']['preprocess']
        masks['hard'][ids] = cv[ids] >= pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= pr['positive_easy_cut'])
        delta = cv[ids]-errors[meta['seed']][0][ids]
        costs = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0)))
        ti, x, ty, td, tpr = bounded.training_data(meta, data)
        tpast = data['geometry'][ti, :16].reshape(-1, 8, 2)
        speed = np.linalg.norm(tpast[:, -1].astype(float)-tpast[:, -2], axis=1)*data['scale'][ti]
        cuts = fit_cuts(td, speed, tpr['known'])
        constant_fraction = (tpr['weights'][:, None]*np.where(tpr['known'][:, None], ty/np.where(td>0, td, 1.)[:, None], 0)).sum(0)
        hpast = data['geometry'][ids, :16].reshape(-1, 8, 2)
        hspeed = np.linalg.norm(hpast[:, -1].astype(float)-hpast[:, -2], axis=1)*data['scale'][ids]
        for arm in ARMS:
            cp = states[key, arm]
            model = build(x.shape[1], cfg['training']['width'], meta['seed'])
            model.load_state_dict(cp['model'])
            train_score = predict(model, x, td, tpr, 'bounded_native')
            for population, score, target, distance, move, support in (
                ('fitting', train_score, ty, td, speed, tpr['known']),
                ('held_source', scores[key, arm], costs, distances[key], hspeed, full[ids])):
                groups = {'all':np.ones(len(distance), bool)}
                groups.update({'disagreement_'+k:v for k,v in strata(distance, cuts['disagreement']).items()})
                groups.update({'speed_'+k:v for k,v in strata(move, cuts['past_step_displacement']).items()})
                for group, mask in groups.items():
                    diagnostics.append(dict(view=key, arm=arm, population=population, stratum=group,
                        **cost_stats(score, target, distance, mask, support, tpr['constant'], constant_fraction)))
            use = full[ids] & chosen[meta['seed']][arm]['strict_stop'][ids]
            quality.append(dict(view=key, arm=arm, selected_complete=int(use.sum()),
                predicted_harm=None if not use.any() else float(scores[key, arm][use, 1].mean()),
                realized_harm=None if not use.any() else float(costs[use, 1].mean())))
        beat(state='fitting_and_held_diagnostics_complete', view=key)
    def metric(a, r, mask=None):
        if mask is None:
            mask = np.ones(n, bool)
        return paired_scene_metrics(a[mask], r[mask], data['sites'][mask], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=cfg['bootstrap_resamples'])
    summaries = {}
    for arm in ARMS:
        summaries[arm] = {}
        for policy in POLICIES:
            seeds, ades, fdes = {}, [], []
            for seed in cfg['seeds']:
                use = chosen[seed][arm][policy]
                ade, fde = np.where(use, errors[seed][0], cv), np.where(use, errors[seed][1], cf)
                ades.append(ade)
                fdes.append(fde)
                seeds[str(seed)] = dict(ADE=metric(ade, cv), FDE=metric(fde, cf), selected=int(use.sum()),
                    selected_unknown=int((use & ~valid.any(1)).sum()), selected_incomplete=int((use & ~full).sum()),
                    zero_CV_harmed=int((ade[masks['zero_CV']] > 0).sum()),
                    full_grid_absolute_gain_bounds={site:[float(np.where(use, bounds[seed][k], 0)[data['sites']==site].mean())
                        for k in ('lower', 'upper')] for site in cfg['sites']},
                    subsets={g:metric(ade, cv, mask) for g,mask in masks.items()})
            summaries[arm][policy] = dict(ADE=metric(np.mean(ades, 0), cv), FDE=metric(np.mean(fdes, 0), cf),
                seeds=seeds, subsets={g:metric(np.mean(ades, 0), cv, mask) for g,mask in masks.items()})
    assert summaries['tempered'] == prior['summaries']['wide_long']
    contrasts = {}
    for policy in POLICIES:
        values = {a:[summaries[a][policy]['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']] for a in ARMS}
        contrasts[policy] = {'tempered_minus_'+a:paired_scene_contrast(values['tempered'], values[a]) for a in ('native', 'fraction')}
    gate = contribution_gate(summaries['tempered']['strict_stop'], contrasts['strict_stop'])
    result = dict(identity=identity, result_source='fresh_24_matched_control_fits_and_readout',
        reference_source='cached_verified_12_tempered_wide_long_heads_and_forecasters',
        summaries=summaries, contrasts=contrasts, archives=archives,
        training=[dict(view=k, arm=a, **{f:r[f] for f in ('fit', 'rows', 'supported_rows', 'checkpoint', 'checkpoint_sha256')})
            for (k,a),r in records.items()], new_updates=sum(r['fit']['step'] for r in records.values()),
        new_draws=sum(r['fit']['total_draws'] for r in records.values()),
        fit_diagnostics=diagnostics, conditional_quality=quality,
        decision_manifest_sha256=file_digest(root/'decisions_complete.json'),
        primary_gates=gate, primary_joint_empirical_pass=all(gate.values()),
        independent_confirmation=False, risk_calibrated=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert result['new_updates'] == 288000 and result['new_draws'] == 73728000
    assert_current(identity)
    immutable_json(public/'analysis.json', result)
    if verify:
        immutable_json(public/'replay.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
            checkpoint_endpoints_replayed=36, fresh_control_endpoints=24, cached_reference_endpoints=12,
            score_rows=n*len(cfg['seeds'])*3, all_checks_passed=True))
    beat(state='verified' if verify else 'evaluated', primary_gates=gate)
