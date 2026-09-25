"""Read out every frozen source-calibrated rule, including failed transport."""
import json
import numpy as np
from scripts import build_m3w_european_selection_data as adapter
from scripts.evaluate_m3w_european_bridge_attribution import seed_summary
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_bridge_risk_calibration import apply, evidence
from src.world_model.m3w_european_source_forecast import baseline_numpy


def verify_done(run, identity):
    path = run.PUBLIC/'completion_checks.json'
    if not path.exists(): return False
    done = json.loads(path.read_text()); assert done['identity'] == identity and done['all_passed']
    for ref in [*done['evaluation_receipts'], done['summary'], done['seed_summary'], done['decisions'], done['calibration']]:
        assert run.artifact(run.ROOT/ref['path']) == ref
    run.beat('cached_verified_complete_readout', new_outcome_computation=False)
    return True


def evaluate(run, cfg, bank, pid, identity):
    freeze = run.PUBLIC/'decision_freeze.json'; run.require_committed(freeze)
    if verify_done(run, identity): return
    frozen = json.loads(freeze.read_text()); assert frozen['identity'] == identity
    ref = frozen['manifest']; assert run.artifact(run.ROOT/ref['path']) == ref
    done = json.loads((run.ROOT/ref['path']).read_text()); assert done['all_passed'] and done['identity'] == identity
    calibrated = run.checked_calibration(identity)
    for ref in done['decisions']:
        assert run.artifact(run.ROOT/ref['path']) == ref
        r = json.loads((run.ROOT/ref['path']).read_text())
        for a in (r['artifact'], r['map']): assert run.artifact(run.ROOT/a['path']) == a
    data = adapter.load(run.parent, pid); labels = adapter.load(run.parent, pid, labels=True)
    sites = data['sites']; n = len(sites); roster = sorted(set(sites)); counters = dict(coordinates=0, metrics=0)
    moving = np.linalg.norm(data['history'][:, -1]-data['history'][:, -2], axis=1) > 0
    def errors(pred):
        result = native_errors(pred, labels['target_eval'], labels['valid'], np.ones(n))
        independent = run.base.cross.independent.coordinate_errors(pred, labels['target_eval'], labels['valid'])
        for u, v in zip(result, independent):
            run.base.cross.independent.close(u, v); counters['coordinates'] += 1
        return result
    classical = [errors(baseline_numpy(data['history'], k)) for k in range(6)]
    cv = classical[1][0]; all_rows = np.ones(n, bool)
    def metric(m, r, mask):
        value = paired_scene_metrics(m[mask], r[mask], sites[mask], expected_scenes=roster,
            dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
            bootstrap_resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
        run.base.cross.independent.check_metric(m, r, sites, mask, value, cfg); counters['metrics'] += 1
        return value
    rows, refs = [], []
    for g in calibrated['groups']:
        name = g['group']; path = run.PUBLIC/'groups'/(name+'.json'); refpath = run.PRIVATE/'evaluation_receipts'/(name+'.json')
        if path.exists():
            assert run.artifact(path) == json.loads(refpath.read_text())
            value = json.loads(path.read_text()); assert value['identity'] == identity and value['verified']
            rows.append(value); refs.append(run.artifact(path)); continue
        start = dict(counters); run.beat('readout_group', group=name)
        pairs, meta = run.attribution.selection(data, bank, g); predictions, action, maps, riskevidence = {}, {}, {}, {}
        endpoints, scorebank = {}, {}
        for pair, (x, env, r, p) in pairs.items():
            ra, rf = errors(r.astype(float)+data['origin'][:, None]); pa, pf = errors(p.astype(float)+data['origin'][:, None])
            endpoints[pair] = (ra, pa)
            values = run.attribution.scores(pair, name)
            for family in cfg['families']:
                prefix = pair+'__'+family; filename = name+'_'+pair+'_'+family
                mapping = json.loads((run.PRIVATE/'maps'/(filename+'.json')).read_text()); maps[prefix] = mapping
                u, m = values[:2] if family == 'neural' else values[2:]; scorebank[prefix] = m
                receipt = json.loads((run.PRIVATE/'decisions'/(filename+'.json')).read_text())
                assert receipt['input_sha256'] == run.array_hash(x) and receipt['score_sha256'] == run.array_hash(u, m)
                with np.load(run.PRIVATE/'decisions'/(filename+'.npz'), allow_pickle=False) as z:
                    for kind, rule in mapping['fitted']['rules'].items():
                        bits = apply(u, m, moving, env, rule); np.testing.assert_array_equal(bits, z[kind])
                        key = prefix+'__'+kind; action[key] = bits
                        predictions[key] = np.where(bits, pa, ra), np.where(bits, pf, rf)
                        riskevidence[key] = evidence(bits, cv, ra, pa, sites, easy_cut=g['easy_cut'])
        full = predictions['full__neural__none']
        masks = dict(all=all_rows, easy=(cv > 0)&(cv <= g['easy_cut']), hard=cv >= g['hard_cut'], complete=labels['valid'].all(1))
        views = {}
        for key, (ade, fde) in predictions.items():
            pair, family, kind = key.split('__'); prefix = pair+'__'+family
            raw = predictions[prefix+'__none']; bits = action[key]; moments = scorebank[prefix]
            reliability = {}
            for site in roster:
                pop = sites == site; known = pop & np.isfinite(ade)
                predicted_den = float(moments[pop, 0].sum()); predicted_harm = float(moments[pop & bits, 1].sum())
                reliability[site] = dict(predicted_reference_mass=predicted_den, predicted_selected_harm=predicted_harm,
                    predicted_ratio=predicted_harm/predicted_den if predicted_den > 0 else None,
                    actual=riskevidence[key]['by_locality'][site],
                    ADE_p95=float(np.quantile(ade[known], .95)), ADE_p99=float(np.quantile(ade[known], .99)))
            views[key] = dict(ADE_vs_raw={s: metric(ade, raw[0], m) for s, m in masks.items()},
                ADE_vs_full_neural={s: metric(ade, full[0], m) for s, m in masks.items()},
                ADE_vs_CV_easy=metric(ade, cv, masks['easy']),
                ADE_vs_training_selected=metric(ade, classical[meta['easy']['baseline_index']][0], all_rows),
                FDE_vs_raw=metric(fde, raw[1], all_rows),
                switch_rate=float(bits.mean()), per_locality=reliability,
                zero_CV_harm=int(((cv == 0)&(ade > 0)).sum()),
                observed_risk_feasible=riskevidence[key]['feasible'], calibration_rule=maps[prefix]['fitted']['rules'][kind],
                calibration_evidence=maps[prefix]['fitted']['evidence'][kind],
                unknown_actions=int((bits & ~np.isfinite(ade)).sum()))
        comparisons = {}
        for pair in cfg['pairs']:
            for family in cfg['families']:
                for kind in ('population_rescale', 'selected_risk_grid'):
                    prefix = pair+'__'+family
                    comparisons[prefix+'__'+kind+'_vs_none'] = (prefix+'__'+kind, prefix+'__none')
            for kind in ('none', 'population_rescale', 'selected_risk_grid'):
                comparisons[pair+'__neural_vs_ridge__'+kind] = (pair+'__neural__'+kind, pair+'__ridge__'+kind)
        contrasts = {k: {s: metric(predictions[a][0], predictions[b][0], mask) for s, mask in masks.items()}
                     for k, (a, b) in comparisons.items()}
        value = dict(identity=identity, group=name, producer=g['producer'], controller=g['controller'],
            seed=int(name.split('_seed')[1].split('_')[0]), views=views, contrasts=contrasts, verified=True,
            checks={k:counters[k]-start[k] for k in counters}, result_source='fresh_run_frozen_calibration_transport',
            role='reused_opened_model_selection', reserved_calibration_opened=False, confirmation_opened=False)
        run.immutable_json(path, value); run.immutable_json(refpath, run.artifact(path)); rows.append(value); refs.append(run.artifact(path))
    run.immutable_json(run.PUBLIC/'seed_averaged_metrics.json', seed_summary(rows, cfg))
    summary = {r['group']: {k: dict(all_gain_vs_raw=v['ADE_vs_raw']['all']['equal_scene_gain_percent'],
        all_gain_vs_full_neural=v['ADE_vs_full_neural']['all']['equal_scene_gain_percent'],
        hard_gain_vs_raw=v['ADE_vs_raw']['hard']['equal_scene_gain_percent'],
        worst_easy_degradation=max(0., -v['ADE_vs_CV_easy']['worst_scene_gain_percent']),
        risk_pass=v['observed_risk_feasible'], switch_rate=v['switch_rate']) for k,v in r['views'].items()} for r in rows}
    run.immutable_json(run.PUBLIC/'summary_metrics.json', dict(groups=summary, independent_localities=roster, rows=n,
        source_C_maps=72, new_neural_training=False, result_source='fresh_run_source_C_fit_and_fixed_readout',
        role='reused_opened_model_selection', calibration_is_source_internal=True,
        reserved_calibration_opened=False, confirmation_opened=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False))
    checks = {k:sum(r['checks'][k] for r in rows)+(12 if k == 'coordinates' else 0) for k in counters}
    run.immutable_json(run.PUBLIC/'completion_checks.json', dict(identity=identity, all_passed=True, independent_checks=checks,
        evaluation_receipts=refs, calibration=run.artifact(run.PUBLIC/'calibration_receipt.json'), decisions=run.artifact(freeze),
        summary=run.artifact(run.PUBLIC/'summary_metrics.json'), seed_summary=run.artifact(run.PUBLIC/'seed_averaged_metrics.json'),
        engineering_completion_not_scientific_success=True))
    run.beat('readout_complete', groups=len(rows), **checks)
