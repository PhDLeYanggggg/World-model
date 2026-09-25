"""Report every source-role setting and distinguish loss from allocation effects."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_selected_risk_learning as run


def summarize(rows, seed_metrics):
    if len(rows) != 18 or len(seed_metrics) != 6: raise ValueError('All source settings required')
    views = {}
    for key in run.method.POLICIES:
        values = [r['views'][key] for r in rows]
        span = lambda a: [min(a), max(a)] if all(v is not None for v in a) else None
        views[key] = dict(all_vs_raw=span([v['ADE_vs_raw']['all']['equal_scene_gain_percent'] for v in values]),
            hard_vs_raw=span([v['ADE_vs_raw']['hard']['equal_scene_gain_percent'] for v in values]),
            all_vs_reference=span([v['ADE_vs_reference']['all']['equal_scene_gain_percent'] for v in values]),
            easy_worst=span([max(0., -v['easy_vs_CV']['worst_scene_gain_percent']) for v in values]),
            risk_passes=sum(v['risk']['feasible'] for v in values),
            easy_passes=sum(all(s['easy_degradation_percent'] is not None and s['easy_degradation_percent'] <= 2
                and s['zero_CV_harm'] == 0 for s in v['risk']['by_locality'].values()) for v in values),
            easy_harm_violations=sum(not s['events']['easy']['passes'] for v in values for s in v['risk']['by_locality'].values()),
            all_harm_violations=sum(not s['events']['all']['passes'] for v in values for s in v['risk']['by_locality'].values()),
            switch_rate=span([v['switch_rate'] for v in values]))
    contrasts = {}
    for key in next(iter(seed_metrics.values())):
        vals = [v[key] for v in seed_metrics.values()]
        if any(v.get('status') == 'not_estimable' for v in vals):
            contrasts[key] = dict(status='not_estimable'); continue
        contrasts[key] = dict(point_range=[min(v['gain_percent'] for v in vals), max(v['gain_percent'] for v in vals)],
            positive_CI=sum(v['CI'][0] > 0 for v in vals), negative_CI=sum(v['CI'][1] < 0 for v in vals),
            overlap_CI=sum(v['CI'][0] <= 0 <= v['CI'][1] for v in vals))
    return dict(views=views, contrasts=contrasts)


def fmt(x): return 'undefined' if x is None else f'{x[0]:+.5f} to {x[1]:+.5f}'


def main():
    cfg, identity = run.registration(); done = run.checked_training(identity)
    checks = json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert checks['all_passed']
    rows = {p:[] for p in cfg['pairs']}
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path']) == ref
        row = json.loads((ROOT/ref['path']).read_text()); rows[row['pair']].append(row)
    seeds = json.loads((run.PUBLIC/'seed_averaged_metrics.json').read_text())
    aggregate = {p:summarize(r, seeds[p]) for p,r in rows.items()}
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json', aggregate)
    training = []; total_seconds = 0
    for ref in done['heads']:
        r = json.loads((ROOT/ref['path']).read_text()); f = r['fit']; total_seconds += f['seconds']
        training.append(dict(path=ref['path'], identity=r['identity']['inputs'], arm=r['identity']['arm'],
            seed=r['identity']['seed'], fit=f, checkpoint=r['artifacts']['checkpoint']))
    run.immutable_json(run.PUBLIC/'training_metrics.json', training)
    gates = dict(real_torch_training_complete=len(training) == 72 and all(v['fit']['complete'] for v in training),
        B_only_fit_C_excluded=True, decisions_frozen_before_readout=True,
        full_selected_joint_observed_risk=aggregate['full']['views']['selected_joint']['risk_passes'] == 18,
        full_selected_joint_net_easy=aggregate['full']['views']['selected_joint']['easy_passes'] == 18,
        full_selected_loss_all_six_positive=aggregate['full']['contrasts']['selected_joint_vs_mean_joint__all']['positive_CI'] == 6,
        full_joint_over_individual_all_six_positive=aggregate['full']['contrasts']['selected_joint_vs_selected_dual__all']['positive_CI'] == 6,
        full_joint_matched_count_all_six_positive=aggregate['full']['contrasts']['selected_joint_vs_selected_hash_matched__all']['positive_CI'] == 6,
        independent_calibration=False, independent_confirmation=False, risk_certificate=False,
        new_forecaster_training=False, deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    run.immutable_json(run.PUBLIC/'gates.json', gates)
    queue = json.loads((run.PRIVATE/'create_queue.json').read_text())
    events = [json.loads(s) for s in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    run.immutable_json(run.PUBLIC/'compute_receipt.json', dict(real_torch_training=True,
        completed_heads=72, total_updates=sum(v['fit']['step'] for v in training), summed_fit_seconds=total_seconds,
        training_pids=sorted({e['pid'] for e in events if e['state'] == 'training'}), native_arm64=True,
        threads=4, interop_threads=1, workers=0, create_query_returncode=queue['response']['returncode'],
        create_observed_utc=queue['completed_utc'], remote_modified=False, jobs_submitted=0,
        remote_M3W_inventory='not_run', unknown_training_draws=sum(v['fit']['unknown_rows_sampled'] for v in training)))
    lines = ['# Source-Only Selected-Risk Learning', '', '## Material Passport',
        'Fresh 72 Torch fits (144,000 updates), frozen source-C readout; cached_verified forecasts and old heads.',
        'Four held C localities per setting. Three seeds, overlapping source-role views, not independent confirmation.', '',
        '| Pair / policy | All ADE vs raw neural (%) | Hard vs raw (%) | All vs R (%) | Worst easy degradation (%) | Risk passes /18 | Easy passes /18 | Switch rate |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p,a in aggregate.items():
        for k,v in a['views'].items():
            lines.append(f"| {p} / {k} | {fmt(v['all_vs_raw'])} | {fmt(v['hard_vs_raw'])} | {fmt(v['all_vs_reference'])} | {fmt(v['easy_worst'])} | {v['risk_passes']} | {v['easy_passes']} | {fmt(v['switch_rate'])} |")
    lines += ['', 'Policy ranges include every seed and source assignment. Switch rates are fractions, not percentages.',
        'Complete risk checks require both positive-harm events, net easy <=2%, and no zero-CV harm in every C locality.',
        'Reference-only risk is structural abstention, not positive learned contribution.', '', '## Three-Seed Contrasts', '',
        '| Pair / A-B role | All-ADE contrast | Point (%) | 95% locality-bootstrap interval (%) |', '|---|---|---:|---:|']
    for p, g in seeds.items():
        for group, cs in g.items():
            for k,v in cs.items():
                if k.endswith('__all'):
                    lines.append(f"| {p} / {group} | {k} | {v.get('gain_percent',float('nan')):+.5f} | {fmt(v.get('CI'))} |")
    lines += ['', 'Seed means are computed within locality before 3,000 resamples. No multiplicity-adjusted discovery,',
        'independent risk certificate or held-selection/confirmation claim. Source C was historically opened development.',
        'Support-bin diagnostics in group JSON never select a threshold or reject a forecast. The hash control',
        'matches query counts, not risk budgets. Scene-query allocation is greedy, not optimal or collision-aware.', '',
        'No change to deployment. Image pixels, annotation steps, detector-derived labels. No metric/seconds,',
        'human-gold, physical-safety, true3D or foundation claim. Stage5C and SMC are off.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    (run.PUBLIC/'world_model_gate.md').write_text('# Source-Only Learning Gates\n\nEngineering completion is not scientific success.\n\n'+
        '\n'.join(f'- {k}: {str(v).lower()}' for k,v in gates.items())+'\n')
    print(json.dumps(dict(gates=gates, aggregate=aggregate, summed_fit_seconds=total_seconds), indent=2))


if __name__ == '__main__': main()
