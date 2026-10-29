"""Lightweight evidence receipt and complete negative paired-training report."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_completion_cost as run
from scripts.report_m3w_forecast_amplitude import risk_causes


def number(x):
    return 'not_estimable' if x is None else f'{x:+.6f}'


def interval(x):
    return 'not_estimable' if x['CI95'] is None else number(x['point'])+' ['+', '.join(number(v) for v in x['CI95'])+']'


def traces():
    rows = []
    for key, ref in sorted(run.models().items()):
        state = run.api.base.read_checkpoint(ROOT/ref['checkpoint']['path'])
        if state['step'] != 2000 or state['trace'][-1]['step'] != 2000:
            raise ValueError('Every complete fixed-step trace required')
        rows.append(dict(context=key[0], seed=key[1], bank=key[2], mode=key[3],
            checkpoint_sha256=ref['checkpoint']['sha256'], steps=state['step'], trace=state['trace'],
            target_columns=state['target_columns'], minibatch_trace_not_validation=True))
    value = dict(result_source='fresh_trace_extraction_cached_verified_complete_Torch_states', heads=72,
        primary_updates=144000, rows=rows, unknown_outcomes_imputed=False,
        comparable_absolute_losses_between_modes=False, new_optimizer_updates=0)
    run.once(run.PUBLIC/'training_traces.json', value)
    return value


def proxy_pool(rows, bank, subset):
    fields = [r['result']['proxy_diagnostics'][bank]['finite_endpoint_proxy_MSE'][subset] for r in rows]
    count = sum(v['real_action_rows'] for v in fields)
    error = sum((np.asarray(v['component_MSE'])*v['real_action_rows'] for v in fields if v['component_MSE'] is not None), np.zeros(5))
    return dict(required_group_views=len(fields), undefined_group_views=sum(v['component_MSE'] is None for v in fields),
        real_action_occurrences=count, normalized_component_MSE=(error/count).tolist() if count else None,
        aggregation='pooled_repeated_real_action_views_not_independent_or_macro_calibration')


def support_ceiling(safety):
    required, undefined = safety['required_record_views'], safety['undefined_easy_record_views']
    if not isinstance(required, int) or not isinstance(undefined, int) or not 0 <= undefined <= required:
        raise ValueError('Valid full-roster support counts required')
    return dict(required_record_views=required, teacher_undefined_easy_record_views=undefined,
        maximum_defined_easy_views_for_nested_policy=required-undefined,
        fixed_head_veto_cannot_pass_complete_support=undefined > 0,
        assumption='same_fixed_floor_easy_reference_and_no_new_selected_rows',
        population_or_arbitrary_model_impossibility_claim=False, future_labels_used_for_inference=False)


def build():
    summary = json.loads((run.PUBLIC/'summary.json').read_text())
    verification = json.loads((run.PUBLIC/'verification.json').read_text())
    complete = json.loads((run.PUBLIC/'complete.json').read_text())
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    if (not verification['exact_replay'] or verification['groups'] != 144 or len(complete['groups']) != 144
            or verification['summary_sha256'] != run.sha(run.PUBLIC/'summary.json')
            or summary['scope'] != 'fitting_resubstitution_only' or summary['held_scored'] or summary['independent_roles_opened']):
        raise ValueError('Complete fitting-only exact readout required')
    rows = []
    for ref in complete['groups']:
        path = ROOT/ref['path']
        if run.sha(path) != ref['sha256']:
            raise ValueError('Completed aggregate group changed')
        rows.append(json.loads(path.read_text()))
    modes = {}
    for mode in run.api.MODES:
        group = [r for r in rows if r['mode'] == mode]
        if len(group) != 72:
            raise ValueError('All original source/context/seed groups required')
        modes[mode] = dict(proxy_fit={bank: {subset: proxy_pool(group, bank, subset)
            for subset in ('all', 'known', 'unknown')} for bank in run.api.BANKS},
            risk_causes={bank: risk_causes([p[bank] for r in group for p in r['result']['recordings'].values()])
                         for bank in run.api.BANKS})
    training = []
    for ref in freeze['fits']:
        path = ROOT/ref['path']
        if run.sha(path) != ref['sha256']:
            raise ValueError('Training receipt changed')
        training.append(json.loads(path.read_text()))
    compute = {key: freeze[key] for key in ('pid', 'seconds', 'count', 'optimizer_updates', 'extra_full_refit_updates',
                                           'pilot_updates', 'peak_RSS_bytes', 'architecture', 'cpu_threads', 'num_workers')}
    compute['primary_checkpoint_bytes'] = sum(r['checkpoint']['bytes'] for r in training)
    compute['supervision_counts'] = {mode: dict(heads=sum(r['mode'] == mode for r in training),
        unknown_row_draws=sum(r['unknown_rows_drawn'] for r in training if r['mode'] == mode),
        unknown_objective_row_visits=sum(r['unknown_rows_with_loss'] for r in training if r['mode'] == mode),
        known_objective_row_visits=sum(r['known_rows_with_loss'] for r in training if r['mode'] == mode)) for mode in run.api.MODES}
    native = {}
    for source in sorted({r['source'] for r in rows}):
        native[source] = {}
        for mode in run.api.MODES:
            group = [r['result'] for r in rows if r['source'] == source and r['mode'] == mode]
            if any(r['known_rows'] <= 0 for r in group):
                raise ValueError('Cannot silently drop unobserved source')
            for name in ('floor', 'physical_neural'):
                values = [(r['full_known_reference_mass'] - r['policies'][name]['policy']['selected_known_benefit_mass']
                           + r['policies'][name]['policy']['selected_known_harm_mass'])/r['known_rows'] for r in group]
                native[source][mode+'_'+name+'_mean_ADE_pixels'] = statistics.fmean(values)
    return dict(result_source='fresh_report_verified_training_and_fitting_readout',
        primary_summary=dict(path=str((run.PUBLIC/'summary.json').relative_to(ROOT)), sha256=run.sha(run.PUBLIC/'summary.json')),
        primary_verification_sha256=run.sha(run.PUBLIC/'verification.json'),
        training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        gates=dict(summary['fitting_gates'], fitting_advancement_gate=summary['fitting_advancement_gate'],
            independent_confirmation=False, deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False),
        compute=compute, readout={key: complete[key] for key in ('pid', 'seconds', 'peak_RSS_bytes')},
        replay=json.loads((run.PRIVATE/'verification_runtime.json').read_text()),
        diagnostics=modes, source_native_diagnostic=native, normalized_losses_are_not_cross_mode_scores=True)


def render(evidence):
    summary = json.loads((run.PUBLIC/'summary.json').read_text()); c = evidence['compute']
    lines = ['# Completion-Aware Cost Supervision: Paired Training Results', '', '## Material Passport', '',
        'Fresh: 72 real Torch cost-head fits, 144,000 primary updates, first full refit,',
        '144 fitting readout groups and separate exact replay. Cached_verified: source',
        'packets, trajectory producers, physical normalizers and static references.',
        'Not_run: new trajectory dynamics, held/independent evaluation or deployment.', '',
        f"Training PID {c['pid']}, exit 0: {c['seconds']:.6f}s; peak RSS {c['peak_RSS_bytes']:,} B.",
        f"Readout PID {evidence['readout']['pid']}: {evidence['readout']['seconds']:.6f}s; separate replay PID {evidence['replay']['pid']}: {evidence['replay']['seconds']:.6f}s.",
        'Native arm64 CPU4, workers0; no NumPy training fallback or resource probing.', '',
        '## Actual Training Intervention', '',
        'Known-only masks unknown-row loss. Completion mode uses explicit lower/upper',
        'cost endpoints, not imputed observations. Original evaluation NaNs are unchanged.',
        'Both modes share initial weights, normalization, source-balanced causal draws,',
        '2,000 steps, architecture and optimizer. The new sampler differs from the older',
        'known-only sampler, so both paired controls are freshly trained.', '',
        '| Mode | Heads | Unknown draws | Unknown objective visits | Known objective visits |',
        '|---|---:|---:|---:|---:|']
    for mode, count in c['supervision_counts'].items():
        lines.append(f"| {mode} | {count['heads']} | {count['unknown_row_draws']} | {count['unknown_objective_row_visits']} | {count['known_objective_row_visits']} |")
    lines += ['', 'These are repeated visits, not independent samples. Extra work: 2,000 full',
        'refit updates and 180 pilot updates. The first complete model/Adam/RNG/trace',
        'refit is exact; checkpoint bytes containing elapsed time need not match.', '',
        '## Full Policy Results', '',
        '| Mode | Policy | Known ADE gain vs floor (%), fitting 95% CI | Required risk support | Undefined easy | Known all/easy violations | Selected (unknown) |',
        '|---|---|---|---:|---:|---:|---:|']
    for mode, data in summary['modes'].items():
        for name, accuracy in data['accuracy'].items():
            s = data['safety'][name]
            lines.append(f"| {mode} | {name} | {interval(accuracy)} | {s['supported_record_views']}/{s['required_record_views']} | "
                f"{s['undefined_easy_record_views']} | {s['known_all_risk_violations']}/{s['known_easy_risk_violations']} | {s['selected']} ({s['selected_unknown']}) |")
    lines += ['', 'Cached static policies and the neural-only head are secondary references,',
        'not new models. Empty selected risk remains undefined; the floor is not credited',
        'as a passed selected-risk estimate. No required recording is dropped.', '',
        '## Completion Minus Matched Known-Only Training', '',
        '| Bank | Full known change (pp), fitting CI | Full completion-lower (pp), fitting CI | Matched completion-lower (pp), fitting CI |',
        '|---|---|---|---|']
    for bank, contrast in summary['completion_minus_known'].items():
        lines.append('| '+bank+' | '+' | '.join(interval(contrast[k]) for k in ('full_known_pp', 'full_lower_pp', 'matched_lower_pp'))+' |')
    lines += ['', 'Count matching concerns utility only. Completion utility uses fixed known',
        'floor-error mass, not an identified complete-population error denominator.', '',
        '## Neural Bank Increment', '', '| Mode | Mixed minus physical known (pp), fitting CI | Completion-lower (pp), fitting CI |',
        '|---|---|---|']
    for mode, data in summary['modes'].items():
        lines.append('| '+mode+' | '+interval(data['neural_bank_increment']['full_known_pp'])+' | '+interval(data['neural_bank_increment']['full_lower_pp'])+' |')
    lines += ['', '## Endpoint-Proxy Fit Diagnostics', '',
        'Pooled normalized squared error over real action occurrences; not training loss,',
        'a source-macro estimate or a conditional-risk certificate. Unknown targets are',
        'interval endpoints, not observed future errors. Undefined group counts remain.', '',
        '| Mode | Bank | Label support | Real actions | Undefined groups | B_lower / H_upper / R_lower / ER_lower / EH_upper MSE |',
        '|---|---|---|---:|---:|---|']
    for mode, d in evidence['diagnostics'].items():
        for bank, pools in d['proxy_fit'].items():
            for subset, p in pools.items():
                values = p['normalized_component_MSE']
                lines.append(f"| {mode} | {bank} | {subset} | {p['real_action_occurrences']} | {p['undefined_group_views']} | "+
                    ('not_estimable' if values is None else ', '.join(f'{v:.6g}' for v in values))+' |')
    lines += ['', '## Per-Locality Reference Sensitivity', '',
        'Primary relative metrics retain every locality. The native values below average',
        'repeated context/head-seed view means in pixels; they are a denominator diagnostic,',
        'not a substitute pooled endpoint, new split or independent observation count.', '',
        '| Locality | Known-only mixed gain (%) | Completion mixed gain (%) | Known-only floor / policy ADE pixels | Completion floor / policy ADE pixels |',
        '|---|---:|---:|---|---|']
    for source, d in evidence['source_native_diagnostic'].items():
        gains = [summary['modes'][mode]['accuracy']['physical_neural']['by_locality'][source] for mode in run.api.MODES]
        values = [f"{d[mode+'_floor_mean_ADE_pixels']:.6g} / {d[mode+'_physical_neural_mean_ADE_pixels']:.6g}" for mode in run.api.MODES]
        lines.append('| '+source+' | '+' | '.join(map(number, gains))+' | '+' | '.join(values)+' |')
    lines += ['', '## Limits And Decision', '',
        'All intervals use 3,000 fitting-locality resamples after averaging repeated',
        'context/head-seed views. They are fitting resubstitution, not independent',
        'generalization, whole-pipeline OOF or three forecasting pipelines. Known FDE',
        'aggregation with any required undefined denominator remains not_estimable.',
        'Supplementary pooled hard/easy values do not replace the full recording gate.', '',
        'Valid interval endpoints do not make a finite fitted cost head calibrated after',
        'selection. This comparison fails the fitting advancement gate. No thresholds,',
        'seeds or checkpoints are selected to hide that result. See failure_analysis.md',
        'and the separate same-region diagnosis; neither changes the original verdict.', '',
        'SOURCE is oblique detector-track midpoint pixels, obs8/pred12/stride12 raw',
        'frames, not SDD human gold, metric/seconds, true 3D or foundation evidence.',
        'Independent roles and DroneCrowd remain closed. No Stage5C or SMC.', '',
        '## Gates', '', *['- '+k+': '+json.dumps(v) for k, v in evidence['gates'].items()], '',
        '[Loss curves](figures/loss_traces.svg), [policy comparison](figures/policy_comparison.svg),',
        '[Chinese reproduction guide](reproduction_zh.md), [training traces](training_traces.json).', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('traces', 'report', 'support'))
    args = parser.parse_args()
    if args.phase == 'traces':
        value = traces(); print(json.dumps(dict(heads=value['heads'], primary_updates=value['primary_updates'])))
    elif args.phase == 'support':
        summary = json.loads((run.PUBLIC/'summary.json').read_text())
        value = support_ceiling(summary['modes']['known_only']['safety']['physical_neural'])
        value['primary_summary_sha256'] = run.sha(run.PUBLIC/'summary.json')
        run.once(run.PUBLIC/'support_ceiling.json', value); print(json.dumps(value))
    else:
        value = build(); run.once(run.PUBLIC/'evidence.json', value); run.once(run.PUBLIC/'gates.json', value['gates'])
        text = render(value); path = run.PUBLIC/'readout.md'
        if path.exists() and path.read_text() != text:
            raise ValueError('Frozen report changed')
        path.write_text(text); print(json.dumps(value['gates']))
