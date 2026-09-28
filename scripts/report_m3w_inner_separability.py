"""Independent source/accounting checks and nominal development statistics."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_inner_separability as run


def complete_mean(values):
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(mean=float(np.mean(valid)) if len(valid) == len(values) and valid else None,
                conditional_mean=float(np.mean(valid)) if valid else None,
                defined=len(valid), undefined=len(values)-len(valid), total=len(values))


def locality_interval(rows, key, draws=3000, seed=20260928):
    sites = sorted({r['site'] for r in rows})
    means = {s: complete_mean([r[key] for r in rows if r['site'] == s]) for s in sites}
    values = [v['mean'] for v in means.values()]
    out = dict(localities=means, summary=complete_mean(values), draws=draws,
               unit='mean_over_dependent_views_within_locality', nominal_only=True)
    if all(v is not None for v in values) and values:
        a = np.array(values)
        boots = a[np.random.default_rng(seed).integers(len(a), size=(draws, len(a)))].mean(1)
        out['CI95'] = np.quantile(boots, [.025, .975]).tolist()
    else:
        out['CI95'] = None
    return out


def ratio_gain(reference, prediction, mask):
    denom = float(reference[mask].sum())
    return 100*(1-float(prediction[mask].sum())/denom) if denom > 0 else None


def assert_value(actual, expected):
    if expected is None:
        assert actual is None
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


def audit_metric(m, cv, floor, neural, cf, ff, nf, take, valid, easycut, hardcut):
    known = np.isfinite(cv)
    selected = np.where(take, neural, floor)
    selected_fde = np.where(take, nf, ff)
    chosen = known & take
    easy = known & (cv > 0) & (cv <= easycut)
    harm = np.maximum(neural-floor, 0)
    values = dict(rows=len(cv), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        intervention_rate=float(take.mean()), known_intervention_rate=float(take[known].mean()),
        unknown_interventions=int(take[~known].sum()), error_sum=float(selected[known].sum()),
        floor_error_sum=float(floor[known].sum()), CV_error_sum=float(cv[known].sum()),
        zero_CV_harmed=int(((cv == 0) & (selected > 0)).sum()))
    for key, at in [('selected_positive_harm_ratio', chosen),
                    ('selected_easy_positive_harm_ratio', chosen & easy)]:
        d = float(floor[at].sum())
        values[key] = float(harm[at].sum())/d if d > 0 else None
    d = float(np.quantile(floor[known], .95))
    values['p95_ratio_to_floor'] = float(np.quantile(selected[known], .95))/d if d > 0 else None
    masks = dict(all=known, easy=easy, hard=known & (cv >= hardcut),
                 complete=known & valid.all(1), partial=known & ~valid.all(1))
    for name, at in masks.items():
        values[name+'_gain_floor'] = ratio_gain(floor, selected, at)
        values[name+'_gain_CV'] = ratio_gain(cv, selected, at)
    at = np.isfinite(cf)
    values['FDE_gain_floor'] = ratio_gain(ff, selected_fde, at)
    values['FDE_gain_CV'] = ratio_gain(cf, selected_fde, at)
    assert set(values) == set(m)
    for key, value in values.items():
        assert_value(m[key], value)
    return len(values)


def risk_coverage(rows, key):
    values = [r['metric'][key] for r in rows]
    known = [v for v in values if v is not None]
    return dict(defined=len(known), undefined=len(values)-len(known),
                violating=sum(v > .02+1e-10 for v in known),
                worst=max(known) if known else None,
                unknown_interventions=sum(r['metric']['unknown_interventions'] for r in rows),
                all_views_pass=bool(known) and len(known) == len(values)
                    and all(v <= .02+1e-10 for v in known)
                    and all(r['metric']['unknown_interventions'] == 0 for r in rows))


def summarize(readout):
    quality = {(r['view'], r['arm']): r for r in readout['quality']}
    metrics = {(r['view'], r['policy']): r for r in readout['rows']}
    views = sorted({r['view'] for r in readout['quality']})
    contrasts = []
    for v in views:
        a, n, initial = [quality[v, k] for k in ('affine', 'nonlinear', 'intercept')]
        am, nm = [metrics[v, k]['metric'] for k in ('affine_matched', 'nonlinear_matched')]
        contrasts.append(dict(view=v, site=a['site'], seed=a['seed'],
            signed_MSE_difference=n['mean_signed_MSE']-a['mean_signed_MSE'],
            gain_MSE_difference=n['signed_MSE'][0]-a['signed_MSE'][0],
            all_risk_MSE_difference=n['signed_MSE'][1]-a['signed_MSE'][1],
            easy_risk_MSE_difference=n['signed_MSE'][2]-a['signed_MSE'][2],
            affine_minus_initial=a['mean_signed_MSE']-initial['mean_signed_MSE'],
            nonlinear_minus_initial=n['mean_signed_MSE']-initial['mean_signed_MSE'],
            matched_ADE_gain_percent=100*(1-nm['error_sum']/am['error_sum']) if am['error_sum'] > 0 else None))
    policies = {}
    for policy in sorted({r['policy'] for r in readout['rows']}):
        rows = [r for r in readout['rows'] if r['policy'] == policy]
        means = {key: complete_mean([r['metric'][key] for r in rows]) for key in rows[0]['metric']}
        policies[policy] = dict(means=means,
            all_risk=risk_coverage(rows, 'selected_positive_harm_ratio'),
            easy_risk=risk_coverage(rows, 'selected_easy_positive_harm_ratio'),
            worst_easy_gain_floor=min(r['metric']['easy_gain_floor'] for r in rows
                                     if r['metric']['easy_gain_floor'] is not None),
            by_site={s: {k: complete_mean([r['metric'][k] for r in rows if r['site'] == s])
                        for k in means} for s in sorted({r['site'] for r in rows})})
    keys = ('signed_MSE_difference', 'matched_ADE_gain_percent',
            'affine_minus_initial', 'nonlinear_minus_initial',
            'gain_MSE_difference', 'all_risk_MSE_difference', 'easy_risk_MSE_difference')
    intervals = {k: locality_interval(contrasts, k) for k in keys}
    return dict(result_source='fresh_run_training_internal_locality_held_readout_and_verification',
        inputs='cached_verified', contrasts=contrasts, intervals=intervals, policies=policies,
        by_seed={str(s): {k: complete_mean([r[k] for r in contrasts if r['seed'] == s])
                         for k in keys} for s in sorted({r['seed'] for r in contrasts})},
        independent_confirmation=False, deployment_changed=False,
        formal_primary_risk_estimand_replaced=False)


def independent_checks(readout, training):
    _, _, data, jobs, oid, _, _, _ = run.old.load()
    causal = {k: data[k] for k in run.old.parent.CAUSAL_KEYS}
    fits = {Path(r['path']).parent.name: r for r in training['groups']}
    metric_rows = {(r['view'], r['policy']): r['metric'] for r in readout['rows']}
    quality = {(r['view'], r['arm']): r for r in readout['quality']}
    counts = dict(training_pairs=0, views=0, metric_values=0, quality_vectors=0,
                  query_count_checks=0, label_occurrences=0, unknown_occurrences=0)
    unique_ids = set()
    slices = []
    loss = {arm: {} for arm in run.api.ARMS}
    for c in run.base.floor_api.contexts(causal, jobs, oid):
        for site in run.sources(c):
            _, ids, x, env, y, pr, ident = run.training_arrays(c, data, site)
            doc = run.checked(fits[c['name']+'_fit_'+site])
            assert doc['identity'] == ident
            states = {}
            for arm, ref in doc['artifacts'].items():
                assert run.base.artifact(ROOT/ref['path']) == ref
                state = run.api.read_checkpoint(ROOT/ref['path'])
                assert state['step'] == 2000 and state['unknown_rows_sampled'] == 0
                run.api.exact(state['preprocess'], pr)
                assert state['identity'] == ident
                states[arm] = state
                for row in state['trace']:
                    loss[arm].setdefault(str(row['step']), []).append(row['monitor'])
            run.api.assert_matched(states['affine'], states['nonlinear'])
            p = [run.api.predict(s, x[:128], env[:128], initial=True)[0] for s in states.values()]
            np.testing.assert_array_equal(*p)
            counts['training_pairs'] += 1
        for pair, (fitting, _) in enumerate(c['pairs']):
            for site in fitting:
                view = c['name']+f'_pair{pair}_from_'+site
                pos, predictions, arrays, meta, pr = run.view_predictions(c, causal, pair, site, fits)
                doc = json.loads((run.PRIVATE/'decisions'/(view+'.json')).read_text())
                assert meta['score_hashes'] == doc['score_hashes']
                run.parent.source.labels.write_or_compare(ROOT/doc['arrays']['path'], arrays, replay=True)
                assert meta['eval_site'] != site and meta['eval_site'] not in meta['outer_held_sites']
                ids = arrays['ids']; unique_ids.update(ids.tolist())
                cv, cf, (floor, ff), (neural, nf) = run.base.floor_api.costs(c, data, pos)
                known = np.isfinite(cv)
                easy = known & (cv > 0) & (cv <= c['job']['design']['easy_cut'])
                counts['label_occurrences'] += int(known.sum())
                counts['unknown_occurrences'] += int((~known).sum())
                groups = {}
                for i, key in enumerate(zip(data['recordings'][ids], data['frames'][ids])):
                    groups.setdefault(key, []).append(i)
                query = [np.array(v) for v in groups.values()]
                for at in query:
                    m = min(int(arrays[a][at].sum()) for a in run.api.ARMS)
                    for a in run.api.ARMS:
                        assert int(arrays[a+'_matched'][at].sum()) == m
                        assert not (arrays[a+'_matched'][at] & ~arrays[a][at]).any()
                    counts['query_count_checks'] += 1
                target = np.column_stack((np.maximum(floor-neural, 0), np.maximum(neural-floor, 0),
                                          floor, np.where(easy, floor, 0),
                                          np.where(easy, np.maximum(neural-floor, 0), 0)))
                target[~known] = np.nan
                local = dict(view=view, site=meta['eval_site'], train_site=site,
                    known=int(known.sum()), supported_known=int((known & arrays['support']).sum()),
                    moving_supported_known=int((known & arrays['support'] & c['moving'][pos]).sum()),
                    evaluation_reference_over_training_mean=float(floor[known].mean()/pr['scale']),
                    description='posthoc_descriptive_no_selection_or_threshold_change', arms={})
                for arm, pred in predictions.items():
                    assert (pred >= 0).all() and np.isfinite(pred).all()
                    assert (pred[:, :2].sum(1) <= c['env'][pos]+1e-5).all()
                    assert (pred[:, 3] <= pred[:, 2]+1e-8).all()
                    assert (pred[:, 4] <= pred[:, 1]+1e-8).all()
                    def signed(z):
                        return np.column_stack((z[:, 0]-z[:, 1], z[:, 1]-.02*z[:, 2], z[:, 4]-.02*z[:, 3]))
                    d = ((signed(pred)-signed(target))/pr['scale']/pr['rms'][5:])**2
                    per_query = [d[at[known[at]]].mean(0) for at in query if known[at].any()]
                    mse = np.mean(per_query, axis=0)
                    np.testing.assert_allclose(quality[view, arm]['signed_MSE'], mse, rtol=2e-11, atol=2e-11)
                    assert_value(quality[view, arm]['mean_signed_MSE'], float(mse.mean()))
                    counts['quality_vectors'] += 1
                    at = known & arrays['support']
                    z = signed(pred)
                    local['arms'][arm] = dict(
                        supported_score_MSE=d[at].mean(0).tolist() if at.any() else None,
                        supported_rows=int(at.sum()),
                        predicted_positive_gain=int((at & (z[:, 0] > 0)).sum()),
                        predicted_all_safe=int((at & (z[:, 1] <= 0)).sum()),
                        predicted_easy_safe=int((at & (z[:, 2] <= 0)).sum()),
                        selected_known=int((known & arrays[arm]).sum()),
                        selected_actual_harmful=int((known & arrays[arm] & (neural > floor)).sum()))
                slices.append(local)
                for (name, policy), row in metric_rows.items():
                    if name != view:
                        continue
                    take = (np.zeros(len(ids), bool) if policy == 'floor' else
                            np.ones(len(ids), bool) if policy == 'neural_unprotected' else arrays[policy])
                    counts['metric_values'] += audit_metric(row, cv, floor, neural, cf, ff, nf, take,
                        data['valid'][ids], c['job']['design']['easy_cut'], c['job']['design']['hard_cut'])
                counts['views'] += 1
        run.beat('independent_verification', **counts)
    assert counts['training_pairs'] == 72 and counts['views'] == 216 and counts['quality_vectors'] == 648
    counts['unique_row_ids'] = len(unique_ids)
    curves = {a: {step: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                   for step, rows in steps.items()} for a, steps in loss.items()}
    return counts, curves, slices


def main():
    started = time.monotonic()
    run.api.torch.set_num_threads(4); run.api.torch.set_num_interop_threads(1)
    _, reg = run.registration()
    assert reg == json.loads((run.PUBLIC/'registration.json').read_text())
    training = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    fit_replay = json.loads((run.PUBLIC/'fit_replay.json').read_text())
    assert fit_replay['exact_except_elapsed'] and fit_replay['heads'] == 2
    for freeze, replay in [('decision_freeze', 'decision_replay')]:
        a, b = [json.loads((run.PUBLIC/(p+'.json')).read_text()) for p in (freeze, replay)]
        assert a['groups'] == b['groups'] and b['exact']
    assert json.loads((run.PUBLIC/'evaluation_replay.json').read_text())['exact']
    readout = json.loads((run.PUBLIC/'readout.json').read_text())
    counts, curves, slices = independent_checks(readout, training)
    summary = summarize(readout)
    summary['verification_counts'] = counts
    summary['training_loss_curves'] = curves
    summary['descriptive_support_diagnostics'] = slices
    run.immutable(run.PUBLIC/'summary.json', summary)
    primary = summary['intervals']['signed_MSE_difference']
    matched = summary['intervals']['matched_ADE_gain_percent']
    p = summary['policies']
    gates = dict(
        real_training_complete=training['heads'] == 144 and training['parameter_updates'] == 288000,
        source_preprocessing_and_replays_checked=counts['training_pairs'] == 72 and counts['views'] == 216,
        primary_development_interval_favors_nonlinear=primary['CI95'] is not None and primary['CI95'][1] < 0,
        matched_ADE_interval_favors_nonlinear=matched['CI95'] is not None and matched['CI95'][0] > 0,
        nonlinear_all_selected_risk_within_budget=p['nonlinear_matched']['all_risk']['all_views_pass'],
        nonlinear_easy_selected_risk_within_budget=p['nonlinear_matched']['easy_risk']['all_views_pass'],
        nonlinear_easy_accuracy_preserved=p['nonlinear_matched']['means']['easy_gain_floor']['undefined'] == 0
            and p['nonlinear_matched']['worst_easy_gain_floor'] >= -2.,
        independent_calibration_complete=False, independent_confirmation_complete=False,
        deployment_promotion=False, stage5c_executed=False, smc_enabled=False)
    run.immutable(run.PUBLIC/'gates.json', gates)
    (run.PUBLIC/'gates.md').write_text('# Engineering and Research Gates\n\n'
        'A completed computation is not a positive research result. Selected-risk gates retain '
        'the original2% budget and fail to establish safety if any view is undefined or contains '
        'unknown-label interventions. No independent calibration or deployment promotion.\n\n'
        +'\n'.join(f'- {key}: `{value}`' for key, value in gates.items())+'\n')
    table = ['| Policy | ADE gain over floor (%) | FDE gain (%) | Easy gain (%) | Hard gain (%) | Intervention (%) | All-risk violations / defined | Undefined all / easy |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v, multiplier=1):
        return 'undefined' if v is None else f'{multiplier*v:.6f}'
    for policy, values in p.items():
        m, risk, erisk = values['means'], values['all_risk'], values['easy_risk']
        columns = [fmt(m[k]['mean'], 100 if k == 'intervention_rate' else 1)
                   for k in ('all_gain_floor', 'FDE_gain_floor', 'easy_gain_floor', 'hard_gain_floor', 'intervention_rate')]
        table.append(f'| {policy} | '+' | '.join(columns)+f" | {risk['violating']}/{risk['defined']} | {risk['undefined']}/{erisk['undefined']} |")
    report = ('# Internal Locality-Held Cost Learning\n\n'
        'Fresh training, causal decisions, internal development readout and independent accounting checks. '
        'Inputs/checkpoint ancestry are cached_verified. Independent confirmation and deployment are not_run.\n\n'
        f"72 unique paired fits,144 heads,288,000 updates; training {training['seconds']:.2f}s. "
        '216 directional views are repeated reuse of those heads, not216 independent experiments. '
        'The evaluated fitting locality contributes no statistics or labels to its head; roles rotate across views. '
        'All12 localities were already development-exposed. Outer-held and independent roles remain closed in this experiment.\n\n'
        '## Registered Primary\n\n'
        f"Nonlinear minus affine normalized signed-score MSE: {primary['summary']['mean']:.8f}; "
        f"nominal95% locality interval {primary['CI95']}. Lower is better. "
        'The affine arm has affine logits and the same nonlinear bounded decoder, not ordinary linear regression.\n\n'
        f"Same-count nonlinear versus affine ADE gain: {matched['summary']['mean']:.6f}%; "
        f"nominal95% interval {matched['CI95']}. Both arms rank only their own admissible pools; "
        'the common count does not make the pools or predicted risks identical.\n\n'
        +'\n'.join(table)+'\n\n'
        'Table percentages are equal-view descriptive means; bootstrap first averages within locality. '
        'A missing denominator stays undefined. Conditional means, unknown-label interventions, per-locality '
        'scores and seed breakdowns are in summary.json; none is silently promoted to a full-roster result.\n\n'
        '## Statistical Scope\n\n'
        '3,000 paired bootstrap draws over12 locality means, after averaging dependent role/seed views. '
        'Three forecaster seeds are reported but share data. Intervals are nominal developmental summaries, '
        'not selection-adjusted or independent confirmation. Accuracy and positive-harm risk are distinct. '
        'The selected-floor risk denominator and2% budget have not changed. Zero intervention is not risk certification.\n\n'
        '## Verification\n\n'
        f"{counts['training_pairs']} training-source preprocess/checkpoint pairs verified; "
        f"{counts['metric_values']:,} independently reduced metric values, {counts['quality_vectors']} "
        f"query-balanced score vectors and {counts['query_count_checks']:,} matched query counts checked. "
        f"{counts['label_occurrences']:,} known and {counts['unknown_occurrences']:,} unknown repeated occurrences; "
        f"{counts['unique_row_ids']:,} unique row IDs, still overlapping and not independent. "
        'One full paired fit replays exactly; all216 decisions and the complete numerical readout replay exactly. '
        'Only the first pair was independently retrained, not all144heads.\n\n'
        '## Scope Limits\n\n'
        'This trains conditional cost heads, not a new world-dynamics forecaster. Observation8/prediction12, '
        'stride12 raw frames, image-local detector-silver. No metric/seconds, human-gold, true3D, '
        'foundation, deployment safety or submission-ready claim. Stage5C/SMC remain off.\n')
    (run.PUBLIC/'results.md').write_text(report)
    checks = ['Simpson: per-locality and seed directions retained, aggregate not substituted for subgroups.',
        'Ecological: no individual-safety inference from locality averages.',
        'Selection:12 already exposed localities; no population-representative claim.',
        'Collider: easy/hard conditioning descriptive, not a causal treatment effect.',
        'Base rates: known/unknown interventions and risk denominators retained.',
        'Regression to mean: all registered views reported, not only worst-source repairs.',
        'Survivorship: all72 fits and216 views retained; undefined scores not zero-filled.',
        'Look elsewhere: secondary metrics nominal, no multiplicity-adjusted significance claim.',
        'Forking paths: committed protocol/final-step training/action freeze; prior development remains exposed.',
        'Causation: capacity contrast concerns this fixed pipeline, not a universal causal mechanism.',
        'Reverse causality: predictions use causal-only whitelist; future outcomes only supervise or evaluate.']
    (run.PUBLIC/'statistical_evidence.md').write_text('# Statistical Evidence and Interpretation\n\n'
        '11/11 interpretation risks checked; this does not mean11 empirical assumptions were proven.\n\n'
        +'\n'.join('- '+s for s in checks)+'\n\n'
        'The12-locality bootstrap does not capture all dependence induced by shared producers and prior research selection. '
        'Intervals describe this developmental comparison; independent calibration and confirmation remain missing.\n')
    tests = ['tests/test_m3w_inner_separability.py', 'tests/test_m3w_inner_separability_report.py',
             'tests/test_m3w_fitting_switch_diagnostic.py', 'tests/test_m3w_query_excess.py']
    test = subprocess.run([sys.executable, '-m', 'pytest', *tests, '-q'], cwd=ROOT, text=True, capture_output=True)
    if test.returncode:
        raise RuntimeError(test.stdout+test.stderr)
    (run.PUBLIC/'scoped_tests.txt').write_text(test.stdout+test.stderr)
    verification = dict(status='verified_internal_development_not_independent_confirmation',
        counts=counts, first_pair_retrained_exact=True, all_decisions_replayed=True, numerical_readout_replayed=True,
        full_legacy_suite='not_run_scoped_tests_only', independent_calibration='not_run_closed',
        source_bindings={**reg['bindings'], str(Path(__file__).relative_to(ROOT)): run.digest(Path(__file__)),
            'tests/test_m3w_inner_separability_report.py': run.digest(ROOT/'tests/test_m3w_inner_separability_report.py')},
        artifacts={p.name: run.digest(p) for p in sorted(run.PUBLIC.iterdir()) if p.is_file() and p.name != 'verification.json'},
        seconds=time.monotonic()-started)
    run.immutable(run.PUBLIC/'verification.json', verification)
    print(json.dumps(dict(primary=primary, matched_ADE=matched, counts=counts, seconds=verification['seconds'])))


if __name__ == '__main__':
    main()
