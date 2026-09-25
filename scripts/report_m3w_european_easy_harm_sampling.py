"""Report all registered sampling settings without selecting a favorable seed."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_easy_harm_sampling as run
from scripts.report_m3w_european_selected_risk_learning import fmt


def span(values):
    if not values or any(v is None or not np.isfinite(v) for v in values):
        return None
    return [float(min(values)), float(max(values))]


def summarize(rows, seeds):
    if len(rows) != 18 or len(seeds) != 6:
        raise ValueError('All 18 role/seed settings and six seed means required')
    views = {}
    for key in run.POLICIES:
        values = [r['views'][key] for r in rows]
        views[key] = dict(
            all_vs_raw=span([v['ADE_vs_raw']['all']['equal_scene_gain_percent'] for v in values]),
            hard_vs_raw=span([v['ADE_vs_raw']['hard']['equal_scene_gain_percent'] for v in values]),
            all_vs_reference=span([v['ADE_vs_reference']['all']['equal_scene_gain_percent'] for v in values]),
            easy_worst=span([max(0., -v['easy_vs_CV']['worst_scene_gain_percent']) for v in values]),
            risk_passes=sum(v['risk']['feasible'] for v in values),
            easy_passes=sum(all(s['easy_degradation_percent'] is not None and
                s['easy_degradation_percent'] <= 2 and s['zero_CV_harm'] == 0
                for s in v['risk']['by_locality'].values()) for v in values),
            easy_harm_violations=sum(not s['events']['easy']['passes']
                for v in values for s in v['risk']['by_locality'].values()),
            all_harm_violations=sum(not s['events']['all']['passes']
                for v in values for s in v['risk']['by_locality'].values()),
            switch_rate=span([v['switch_rate'] for v in values]))
    contrasts = {}
    for key in next(iter(seeds.values())):
        values = [g[key] for g in seeds.values()]
        if any(v.get('status') == 'not_estimable' for v in values):
            contrasts[key] = dict(status='not_estimable'); continue
        contrasts[key] = dict(point_range=span([v['gain_percent'] for v in values]),
            positive_CI=sum(v['CI'][0] > 0 for v in values),
            negative_CI=sum(v['CI'][1] < 0 for v in values),
            overlap_CI=sum(v['CI'][0] <= 0 <= v['CI'][1] for v in values))
    return dict(views=views, contrasts=contrasts)


def diagnostics(training, rows):
    if len(training) != 36:
        raise ValueError('Thirty-six completed matched fits required')
    out = {}
    for pair in ('full', 'motion_only'):
        fits = [r for r in training if r['pair'] == pair]
        assert len(fits) == 18
        ratio = [r['fit']['trace'][-1]['moment_mse']/r['control_fit']['trace'][-1]['moment_mse'] for r in fits]
        masses = []
        for row in rows[pair]:
            for site, value in row['query_budget'].items():
                e = value['events']['easy']; actual = e['actual_selected_harm']
                if actual > 0:
                    masses.append(dict(group=row['group'], site=site,
                        predicted_over_actual=e['predicted_selected_harm_supported']/actual,
                        actual_ratio=e['actual_harm_ratio'], predicted_ratio=e['predicted_harm_ratio'],
                        unknown_mass_fraction=e['unknown_predicted_mass_fraction']))
        out[pair] = dict(
            fixed_batch_mse_ratio_range=span(ratio), fixed_batch_mse_ratio_median=float(np.median(ratio)),
            fixed_batch_mse_lower_count=sum(r < 1 for r in ratio),
            base_positive_probability=span([r['fit']['base_positive_probability'] for r in fits]),
            sampled_positive_probability=span([r['fit']['sampled_positive_probability'] for r in fits]),
            empirical_positive_draw_fraction=span([r['fit']['positive_easy_harm_draws']/r['fit']['total_draws'] for r in fits]),
            selected_easy_harm_supported_ratio_median=float(np.median([r['predicted_over_actual'] for r in masses])) if masses else None,
            selected_easy_harm_underpredicted_views=sum(r['predicted_over_actual'] < 1 for r in masses),
            selected_easy_harm_positive_views=len(masses), mass_views=masses)
    return dict(pairs=out, diagnostic_batch_not_validation=True,
        weighted_batch_loss_not_paired_uniform_diagnostic=True,
        differing_selected_populations_not_calibration_ablation=True,
        result_source='fresh_run_descriptive_diagnostics', decisions_changed=False)


def main():
    cfg, identity = run.registration(); done = run.checked_training(identity)
    checked = json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert checked['all_passed']
    rows = {p:[] for p in cfg['pairs']}
    for ref in checked['groups']:
        assert run.artifact(ROOT/ref['path']) == ref
        row = json.loads((ROOT/ref['path']).read_text()); rows[row['pair']].append(row)
    assert run.artifact(ROOT/checked['seeds']['path']) == checked['seeds']
    seeds = json.loads((run.PUBLIC/'seed_averaged_metrics.json').read_text())
    aggregate = {p:summarize(v, seeds[p]) for p,v in rows.items()}
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json', aggregate)
    fits = []
    for ref in done['heads']:
        r = json.loads((ROOT/ref['path']).read_text()); input_ref = r['identity']['input']
        receipt = json.loads((ROOT/input_ref['path']).read_text())
        inputs = json.loads((ROOT/receipt['parent_inputs']['path']).read_text())
        old = json.loads((ROOT/receipt['uniform_control']['path']).read_text())
        fits.append(dict(group=inputs['group']['group'], pair=inputs['pair'], fit=r['fit'],
            control_fit=old['fit'], checkpoint=r['artifacts']['checkpoint'], control=receipt['uniform_control']))
    run.immutable_json(run.PUBLIC/'training_metrics.json', fits)
    run.immutable_json(run.PUBLIC/'diagnostic_summary.json', diagnostics(fits, rows))
    v = aggregate['full']['views']; c = aggregate['full']['contrasts']
    gates = dict(real_torch_training_complete=len(fits)==36 and all(r['fit']['complete'] for r in fits),
        original_expected_objective_preserved=True, B_only_fit=True, decisions_frozen_before_readout=True,
        corrected_joint_complete_observed_risk=v['corrected_joint']['risk_passes']==18,
        corrected_joint_net_easy=v['corrected_joint']['easy_passes']==18,
        corrected_joint_all_six_positive_vs_mean=c['corrected_joint_vs_mean_joint__all']['positive_CI']==6,
        corrected_joint_all_six_positive_vs_raw=c['corrected_joint_vs_raw_neural__all']['positive_CI']==6,
        joint_all_six_positive_vs_individual=c['corrected_joint_vs_corrected_dual__all']['positive_CI']==6,
        joint_all_six_positive_vs_matched_count=c['corrected_joint_vs_corrected_hash_matched__all']['positive_CI']==6,
        independent_calibration=False, independent_confirmation=False, risk_certificate=False,
        new_forecaster_training=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)
    run.immutable_json(run.PUBLIC/'gates.json', gates)
    queue = json.loads((run.PRIVATE/'create_queue.json').read_text())
    events = [json.loads(s) for s in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    run.immutable_json(run.PUBLIC/'compute_receipt.json', dict(completed_heads=len(fits),
        total_updates=sum(r['fit']['step'] for r in fits),
        summed_fit_seconds=sum(r['fit']['seconds'] for r in fits), native_arm64=True, real_torch_training=True,
        training_pids=sorted({e['pid'] for e in events if e['state']=='training'}),
        threads=4, interop_threads=1, workers=0,
        unknown_training_draws=sum(r['fit']['unknown_rows_sampled'] for r in fits),
        create_query_returncode=queue['response']['returncode'], create_observed_utc=queue['completed_utc'],
        remote_modified=False, jobs_submitted=0, remote_M3W_inventory='not_run'))
    lines = ['# Importance-Corrected Easy-Harm Sampling', '',
        'Fresh 36 Torch fits / 72,000 updates and 180 new policy views; 324 old views cached_verified.',
        'The 504 views cover three seeds and six ordered source assignments in each of two forecast pairs.',
        'Source C is excluded from the current A/B fitted chain but historically opened development.', '',
        '| Pair / policy | All ADE vs raw neural (%) | Hard vs raw (%) | All vs R (%) | Worst easy degradation (%) | Risk /18 | Net easy /18 | Switch fraction |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p,a in aggregate.items():
        for key,v in a['views'].items():
            lines.append(f"| {p} / {key} | {fmt(v['all_vs_raw'])} | {fmt(v['hard_vs_raw'])} | {fmt(v['all_vs_reference'])} | {fmt(v['easy_worst'])} | {v['risk_passes']} | {v['easy_passes']} | {fmt(v['switch_rate'])} |")
    lines += ['', 'Complete risk requires both positive-harm ratios <=2%, net easy degradation <=2%,',
        'and no zero-CV harm, in every supported C locality. Net easy passing is not complete risk passing.',
        'Reference-only abstention is not learned improvement. No deployment selection is made.', '',
        '## Three-Seed Comparisons', '',
        '| Pair / A-B roles | All-ADE contrast | Point (%) | 95% locality-bootstrap CI (%) |', '|---|---|---:|---:|']
    for pair,groups in seeds.items():
        for group,values in groups.items():
            for key,value in values.items():
                if key.endswith('__all'):
                    lines.append(f"| {pair} / {group} | {key} | {value.get('gain_percent',float('nan')):+.5f} | {fmt(value.get('CI'))} |")
    lines += ['', 'Seeds are averaged within locality, then four C localities are resampled 3,000 times.',
        'Overlapping source-role views are not independent replications. No multiplicity-adjusted discovery.',
        'The exact expected-loss identity does not make Adam updates unbiased, nor guarantee lower variance.',
        'Greedy query allocation is not collision-aware or optimal. Matched hash controls query counts, not risk.', '',
        '8 observed / 12 predicted annotation steps, raw stride 12; image pixels, detector-derived labels.',
        'No metric, seconds-level, human-gold, physical-safety, true3D, foundation or submission-ready claim.',
        'Selection, reserved calibration and confirmation are not read this round. Stage5C and SMC remain off.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    (run.PUBLIC/'world_model_gate.md').write_text('# Sampling Experiment Gates\n\nEngineering completion is not research success.\n\n'+
        '\n'.join(f'- {k}: {str(v).lower()}' for k,v in gates.items())+'\n')
    print(json.dumps(dict(gates=gates, aggregate=aggregate), indent=2))


if __name__ == '__main__': main()
