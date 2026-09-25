"""Summarize fixed attribution contrasts without selecting a deployment winner."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_bridge_attribution as run


def summarize(rows, seeds):
    if len(rows) != 18 or len(seeds) != 6: raise ValueError('All registered groups required')
    names = set(rows[0]['views']); views = {}
    if any(set(r['views']) != names for r in rows): raise ValueError('Missing control')
    for name in sorted(names):
        v = [r['views'][name] for r in rows]
        def bounds(values):
            return [min(values), max(values)] if all(x is not None for x in values) else None
        views[name] = dict(
            all_vs_full_neural=bounds([x['ADE_vs_full_neural']['all']['equal_scene_gain_percent'] for x in v]),
            hard_vs_full_neural=bounds([x['ADE_vs_full_neural']['hard']['equal_scene_gain_percent'] for x in v]),
            all_vs_training_selected=bounds([x['ADE_vs_training_selected']['equal_scene_gain_percent'] for x in v]),
            easy_degradation=bounds([x['worst_easy_degradation'] for x in v]),
            easy_pass=sum(bool(x['easy_pass']) for x in v),
            switch_rate=bounds([x['switch_rate'] for x in v]),
            realized_easy_risk_exceedances=sum(
                site['positive_harm']['easy']['realized_ratio'] is not None and site['positive_harm']['easy']['realized_ratio'] > .02
                for x in v for site in x['per_locality'].values()),
            locality_views_are_dependent=True)
    contrasts = {}
    for key in next(iter(seeds.values())):
        vals = [v[key] for v in seeds.values()]
        if any(v.get('status') == 'not_estimable' for v in vals):
            contrasts[key] = dict(status='not_estimable'); continue
        contrasts[key] = dict(point_range=[min(v['gain_percent'] for v in vals), max(v['gain_percent'] for v in vals)],
            positive_CI=sum(v['CI'][0] > 0 for v in vals), negative_CI=sum(v['CI'][1] < 0 for v in vals),
            overlapping_CI=sum(v['CI'][0] <= 0 <= v['CI'][1] for v in vals))
    return dict(views=views, contrasts=contrasts)


def gates(aggregate):
    c, v = aggregate['contrasts'], aggregate['views']
    return dict(real_motion_only_retraining_complete=True, no_future_or_role_leakage_checked=True,
        full_cached_action_replay_exact=True, per_query_action_counts_matched=True,
        full_neural_beats_ridge_all_six=c['full__neural_vs_ridge__all']['positive_CI'] == 6,
        full_neural_matched_ranking_beats_ridge_all_six=c['full__neural_matched_vs_ridge_matched__all']['positive_CI'] == 6,
        neural_trajectory_candidates_help_all_six=c['full_neural_vs_motion_neural__all']['positive_CI'] == 6,
        full_neural_net_easy_preserved=v['full__neural']['easy_pass'] == 18,
        motion_only_neural_net_easy_preserved=v['motion_only__neural']['easy_pass'] == 18,
        independent_calibration=False, independent_confirmation=False, certified_positive_harm=False,
        new_neural_dynamics_training=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)


def fmt(v):
    return 'undefined' if v is None else f'{v[0]:+.6f}% to {v[1]:+.6f}%'


def main():
    done = json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert done['all_passed']
    rows = []
    for ref in done['evaluation_receipts']:
        assert run.artifact(ROOT/ref['path']) == ref
        rows.append(json.loads((ROOT/ref['path']).read_text()))
    seeds = json.loads((run.PUBLIC/'seed_averaged_metrics.json').read_text())
    aggregate = summarize(rows, seeds); gate = gates(aggregate)
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json', aggregate)
    run.immutable_json(run.PUBLIC/'gates.json', gate)
    lines = ['# Bridge Attribution Results', '',
        'Fresh motion-only cost training; full-pair fits and inputs cached_verified; fresh fixed readout.',
        'Six reused opened model-selection localities, not independent confirmation. Three seeds;',
        '18 dependent producer/controller/seed configurations; 396 policy views. No selected winner.', '',
        'All gains below compare delivered predictions with the preceding full neural all-risk bridge.', '',
        '| Pair and scoring policy | All gain | Hard gain | Gain vs training-selected motion | Easy degradation | Easy passes |',
        '|---|---:|---:|---:|---:|---:|']
    for k, v in aggregate['views'].items():
        lines.append(f"| {k} | {fmt(v['all_vs_full_neural'])} | {fmt(v['hard_vs_full_neural'])} | {fmt(v['all_vs_training_selected'])} | {fmt(v['easy_degradation'])} | {v['easy_pass']}/18 |")
    lines += ['', 'Ranges cover all configurations, not best seeds. Easy means worst-locality net',
        'degradation relative to causal CV on the producer-training-defined positive-easy event.',
        'Positive harm is separate and remains uncalibrated. Full FDE, complete-label, p95/p99,',
        'per-locality errors and predicted/realized moment accounting are retained in group JSON.', '',
        '## Primary Three-Seed Contrasts', '',
        'Seeds are averaged within locality before 3,000 locality resamples. All six localities',
        'remain the independent unit. Intervals are exploratory, without multiplicity adjustment.', '',
        '| Producer/controller | Contrast | All gain | 95% locality CI | Hard gain |',
        '|---|---|---:|---:|---:|']
    primary = ('full__neural_vs_ridge', 'full__neural_matched_vs_ridge_matched',
        'motion_only__neural_vs_ridge', 'motion_only__neural_matched_vs_ridge_matched', 'full_neural_vs_motion_neural')
    for group, comparisons in seeds.items():
        for key in primary:
            allv, hard = comparisons[key+'__all'], comparisons[key+'__hard']
            lines.append(f"| {group} | {key} | {allv['gain_percent']:+.6f}% | {fmt(allv['CI'])} | {hard['gain_percent']:+.6f}% |")
    lines += ['', '## Matched Query Coverage', '',
        '| Group | Pair | Queries | Zero-count queries | Matched actions | Different rows | Different queries |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        for pair, v in r['coverage'].items():
            lines.append(f"| {r['group']} | {pair} | {v['queries']} | {v['zero_count_queries']} | {v['matched_actions']} | {v['matched_disagree_rows']} | {v['matched_disagree_queries']} |")
    lines += ['', 'Matched counts include unknown-label rows. They are not a common estimated risk',
        'budget: each scorer has its own moments; hash ranking is not necessarily risk-guarded.',
        'Motion-only excludes neural trajectories and policy bits, not learned floor/cost scorers.',
        'Neural risk uses hurdle/ranking losses and ridge uses moment regression; their contrast',
        'does not isolate architecture nonlinearity. Full vs motion-only changes the forecast pair.',
        'Image-pixel native annotation-step results only. No physical-safety or deployment claim.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    (run.PUBLIC/'world_model_gate.md').write_text('# Bridge Attribution Gates\n\nEngineering completion is not hypothesis confirmation.\n\n'+
        '\n'.join(f'- {k}: {str(v).lower()}' for k, v in gate.items())+'\n')
    training = json.loads((run.PRIVATE/'training_complete.json').read_text()); records = []
    for ref in training['heads']:
        r = json.loads((ROOT/ref['path']).read_text()); fit = r['fit']
        records.append(dict(group=r['identity']['group'], task=r['identity']['task'], parameters=fit['parameters'],
            step=fit['step'], seconds=fit['seconds'], unknown_rows_sampled=fit['unknown_rows_sampled'],
            trace=fit['trace'], fixed_trace=fit.get('fixed_trace'), checkpoint=r['artifacts']['checkpoint']))
    run.immutable_json(run.PUBLIC/'training_metrics.json', dict(heads=records, updates=72000,
        total_head_training_seconds=sum(r['seconds'] for r in records),
        loss_scale='source CV scale; not raw FDE; minibatch losses need not be monotone',
        real_torch_training=True, new_forecaster_training=False, pilot_in_fixed_budget=True))
    queue = json.loads((run.PRIVATE/'create_queue.json').read_text())
    events = [json.loads(v) for v in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    run.immutable_json(run.PUBLIC/'compute_receipt.json', dict(runtime='native_arm64_torch_cpu', threads=4,
        interop_threads=1, dataloader_workers=0, new_heads=36, new_ridge=36, new_updates=72000,
        cached_full_heads=36, cached_full_ridge=36, checkpoint_every=200, heartbeat_every=200,
        training_pids=sorted({v['pid'] for v in events if v['state'] == 'head_started'}),
        head_training_seconds=sum(r['seconds'] for r in records), pilot=run.artifact(run.PRIVATE/'pilot.json'),
        create_queue_returncode=queue['response']['returncode'], create_checked_at=queue['completed_utc'],
        create_jobs_submitted=0, remote_modified=False, remote_project_inventory='not_run',
        placement='Source data local; actual pilot confirmed CPU4 full cost-head budget feasible'))
    print(json.dumps(dict(contrasts=aggregate['contrasts'], gates=gate), indent=2))


if __name__ == '__main__': main()
