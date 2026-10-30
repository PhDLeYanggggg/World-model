"""Verified matched active-bank evidence, actual training curves and aggregates."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_active_bank_cost as run
from src.evaluation import m3w_metric_bundle as bundle


def interval(value):
    if value['CI95'] is None:
        return 'not_estimable'
    return f"{value['point']:+.6f} [{value['CI95'][0]:+.6f}, {value['CI95'][1]:+.6f}]"


def support_profile(complete):
    sites, records, keys = {}, set(), set()
    rows = known = unknown = 0
    for ref in complete['groups']:
        path = ROOT / ref['path']
        if run.sha(path) != ref['sha256']:
            raise ValueError('Frozen cost aggregate changed')
        row = json.loads(path.read_text())
        key = row['context'], row['source'], row['seed']
        if key in keys:
            raise ValueError('Repeated source/context/seed readout')
        keys.add(key)
        r = row['result']
        current = dict(rows=r['rows'], known=r['known_rows'], unknown=r['unknown_rows'], ids_sha256=row['ids_sha256'])
        if row['source'] in sites and sites[row['source']] != current:
            raise ValueError('Repeated fitting source has changed query support')
        sites[row['source']] = current
        records.update(r['recordings'])
        rows += r['rows']
        known += r['known_rows']
        unknown += r['unknown_rows']
    if len(keys) != 72 or len(sites) != 12:
        raise ValueError('All72groups and12localities required')
    return dict(unique_recordings=len(records), unique_queries=sum(v['rows'] for v in sites.values()),
        unique_known_queries=sum(v['known'] for v in sites.values()),
        unique_unknown_queries=sum(v['unknown'] for v in sites.values()),
        repeated_row_views=rows, repeated_known_views=known, repeated_unknown_views=unknown,
        by_locality=sites, repeated_views_not_independent_samples=True)


def freeze(value):
    path = run.PUBLIC / 'evidence.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if dict(value, replay=previous['replay']) != previous:
            raise ValueError('Frozen matched-bank artifact differs')
        return previous
    run.once(path, value)
    return value


def evidence():
    check = json.loads((run.PUBLIC / 'verification.json').read_text())
    if (not check['all72groups_exact'] or not check['all_old_predictions_actions_record_scores_exact']
            or check['summary_sha256'] != run.sha(run.PUBLIC / 'summary.json')
            or check['complete_sha256'] != run.sha(run.PUBLIC / 'complete.json')):
        raise ValueError('Complete separate exact policy inference replay required')
    models = run.models()
    traces = []
    for (context, seed, variant), row in sorted(models.items()):
        state = run.api.completion.base.read_checkpoint(ROOT / row['checkpoint']['path'])
        if state['step'] != 2000 or state['trace'][-1]['step'] != 2000 or state['unknown_rows_with_loss']:
            raise ValueError('Complete actual known-only cost-head traces required')
        traces.append(dict(context=context, seed=seed, variant=variant, trace=state['trace'],
            checkpoint_sha256=row['checkpoint']['sha256'], minibatch_not_validation=True))
    run.once(run.PUBLIC / 'training_traces.json', dict(rows=traces, heads=54, primary_optimizer_updates=108000,
        result_source='cached_verified_actual_completed_Torch_cost_training_traces'))
    loss = {v: dict(models=18, first_median=float(np.median([r['trace'][0]['loss'] for r in traces if r['variant'] == v])),
        final_median=float(np.median([r['trace'][-1]['loss'] for r in traces if r['variant'] == v]))) for v in run.api.VARIANTS}
    training = json.loads((run.PUBLIC / 'training_freeze.json').read_text())
    complete = json.loads((run.PUBLIC / 'complete.json').read_text())
    summary = json.loads((run.PUBLIC / 'summary.json').read_text())
    value = dict(result_source='verified_fresh_matched_Torch_cost_experiment_aggregate_receipt',
        training={k: training[k] for k in ('pid', 'seconds', 'peak_RSS_bytes', 'count', 'primary_optimizer_updates',
            'extra_full_refit_updates', 'pilot_updates', 'legacy_exact_endpoints')},
        readout={k: complete[k] for k in ('pid', 'seconds', 'peak_RSS_bytes')},
        replay=json.loads((run.PRIVATE / 'verification_runtime.json').read_text()),
        checkpoint_bytes=sum(r['checkpoint']['bytes'] for r in models.values()),
        loss_summary=loss, support_profile=support_profile(complete),
        summary_sha256=run.sha(run.PUBLIC / 'summary.json'), verification_sha256=run.sha(run.PUBLIC / 'verification.json'),
        artifact_code_bindings={str(p.relative_to(ROOT)): run.sha(p) for p in
            (Path(__file__), ROOT / 'tests/test_m3w_active_bank_artifacts.py')},
        fitting_gates=summary['fitting_gates'], fitting_advancement_gate=summary['fitting_advancement_gate'],
        held_scored=False, independent_roles_opened=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, metric_claim=False, seconds_claim=False,
        stage5c_executed=False, smc_enabled=False)
    return freeze(value), summary


def render(e, s):
    t, p = e['training'], e['support_profile']
    text = ['# Gain/Harm Learning On Matched Frozen Active Banks', '', '## Material Passport', '',
        'Fresh_run: 54 real Torch cost heads, 108,000 primary updates, 2,000 full-',
        'refit updates, 160 pilot updates and a complete separate policy replay.',
        'Cached_verified: original SOURCE inputs, original cost controls and frozen',
        'seed43 trajectory producers. Three new past-only proposal caches were',
        'built in this round; they are frozen inputs, not new trajectory training.',
        'Not_run: controller-held outcomes, independent calibration/confirmation,',
        'new scene-joint decision training or deployment.', '',
        f"Training PID{t['pid']}: {t['seconds']:.6f}s, peak RSS {t['peak_RSS_bytes']:,}B.",
        f"Readout PID{e['readout']['pid']}: {e['readout']['seconds']:.6f}s; separate exact replay PID{e['replay']['pid']}: {e['replay']['seconds']:.6f}s.",
        'All 36 old cost endpoints reproduce their full scientific states. Old',
        'moments, actions and required-recording risk scores replay exactly in all',
        '72 groups. The first new-bank head has an exact 2,000-step independent',
        'refit. Native arm64 CPU4/workers0; no NumPy training fallback or budget cut.', '',
        '## Same Learner, One Replaced Candidate', '',
        'All six physical candidates, the causal floor, original past-only coverage',
        'cohort, draws and physical/detector fitting preprocessing stay fixed.',
        'Only the seventh candidate changes. Its motion/region features, envelope',
        'and supervised costs are rebuilt together through separate causal/label',
        'interfaces. Cross-bank comparisons use actual paths, not stale action6 costs.', '',
        f"Unique fitting queries: {p['unique_queries']:,}; recordings: {p['unique_recordings']}; localities: 12.",
        f"Known: {p['unique_known_queries']:,}; wholly unknown: {p['unique_unknown_queries']:,}; repeated row views: {p['repeated_row_views']:,}.",
        'Repeated windows/context/seed views are not independent samples. Intervals',
        'use 3,000 locality resamples after context/head-seed averaging. SOURCE has',
        'historical development exposure: these are fitting outcomes, not a new test.', '',
        '## Complete-Policy Accuracy And Risk', '',
        '| Bank | Known ADE gain vs floor %, CI95 | FDE gain %, CI95 | Recording risk supported | Known easy-risk violations | Undefined easy-risk views | Switch row views |',
        '|---|---|---|---:|---:|---:|---:|']
    for v in run.api.VARIANTS:
        risk = s['safety'][v]
        text.append(f"| {v} | {interval(s['accuracy'][v])} | {interval(s['FDE'][v])} | {risk['supported_record_views']}/{risk['required_record_views']} | {risk['known_easy_risk_violations']} | {risk['undefined_easy_record_views']} | {risk['selected']} |")
    text += ['', 'Risk support uses the unchanged 2% complete-policy all/easy rule on',
        'every required recording, including missing-outcome bounds. Raw easy mean',
        'gain and positive selected-easy risk are different quantities. Empty risks',
        'are not_estimable, not safe zero. Complete-roster FDE stays undefined when',
        'a required locality has no requested final labels.', '',
        '## Paired Candidate-Bank Contribution', '',
        '| Contrast | Full known pp, CI95 | Full completion-lower pp, CI95 | Full completion-upper pp, CI95 | Same-count lower pp, CI95 |',
        '|---|---|---|---|---|']
    for label, values in s['comparisons'].items():
        text.append('| ' + label + ' | ' + ' | '.join(interval(values[k]) for k in
            ('full_known_pp', 'full_lower_pp', 'full_upper_pp', 'matched_lower_pp')) + ' |')
    text += ['', 'Differences retain the same known floor-reference mass, not a newly',
        'selected denominator. Negative completion bounds are not observed damage.',
        'Same-count controls assess utility only. Empty comparisons retain undefined',
        'points/intervals and are not credited as passing safety controls.', '',
        '## Does The New Head Actually Use Neural Trajectories?', '',
        'The same frozen new head is also decoded with its neural action disabled,',
        'without refitting. This secondary control separates neural trajectory use',
        'from changes in physical-action decisions due to shared-head training.', '',
        '- Neural-selected repeated row views: ' + str(s['new_head_neural_usage_counts']['neural_selected']),
        '- Unknown neural-selected views: ' + str(s['new_head_neural_usage_counts']['unknown_neural_selected']),
        '- Full known neural-use increment pp: ' + interval(s['new_head_neural_usage_increment']['full_known_pp']),
        '- Full completion-lower neural-use increment pp: ' + interval(s['new_head_neural_usage_increment']['full_lower_pp']), '',
        '## Future-Informed Opportunity, Not A Learned Policy', '',
        '| Bank | Required context/record views, seeds deduplicated | Feasible | Positive feasible |',
        '|---|---:|---:|---:|']
    for v, o in s['bank_opportunity_diagnostic'].items():
        text.append(f"| {v} | {o['required_record_views']} | {o['feasible_record_views']} | {o['positive_record_views']} |")
    text += ['', 'The shared union-oracle regret is diagnostic and has no inference role.',
        'The union is not an eight-action deployment bank. Bank-local regrets are',
        'preserved for legacy equality, not compared as one common oracle.', '',
        '## Real Training Objective', '',
        '| Bank | Heads | First median minibatch objective | Final median minibatch objective |',
        '|---|---:|---:|---:|']
    for v, loss in e['loss_summary'].items():
        text.append(f"| {v} | {loss['models']} | {loss['first_median']:.6g} | {loss['final_median']:.6g} |")
    text += ['', 'Loss is asymmetric five-moment cost regression plus within-row ranking,',
        'not best-class classification. Curves are actual fitting minibatches, not',
        'validation or independent-test losses. Unknown labels receive no observed-',
        'cost loss and do not determine the common row sampler.', '', '## Fitting-Progress Gates', '',
        *['- ' + k + ': ' + json.dumps(v) for k, v in s['fitting_gates'].items()],
        '- fitting_advancement_gate: ' + json.dumps(s['fitting_advancement_gate']),
        '- independent_confirmation: false', '- deployment_changed: false', '- submission_ready: false', '',
        'SOURCE is supplied detector-silver bbox-midpoint pixels, obs8/pred12 at',
        'stride12 raw frames. Not historical SDD t50, human gold, verified live',
        'sensor-as-of, metric/seconds, true3D or foundation evidence. No Stage5C',
        'execution or SMC. A complete anonymous raw-data/weight release and the',
        'scene-joint/independent-evidence requirements are not fulfilled here.', '',
        '[Actual curves](figures/summary.svg), [machine evidence](evidence.json),',
        '[failure analysis](failure_analysis.md), [model/data card](model_data_card.md),',
        '[Chinese guide](reproduction_zh.md).', '']
    return '\n'.join(text)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    check = json.loads((run.PUBLIC / 'verification.json').read_text())
    if not check['all72groups_exact'] or check['summary_sha256'] != run.sha(run.PUBLIC / 'summary.json'):
        raise ValueError('Verified complete matched-bank result required')
    s = json.loads((run.PUBLIC / 'summary.json').read_text())
    traces = json.loads((run.PUBLIC / 'training_traces.json').read_text())['rows']
    plt.rcParams.update({'font.size': 9, 'svg.hashsalt': run.NAME, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ['#666666', '#a85e36', '#227d99']
    for i, v in enumerate(run.api.VARIANTS):
        point = s['accuracy'][v]
        if point['CI95'] is not None:
            axes[0].errorbar(i, point['point'], yerr=np.array([[point['point'] - point['CI95'][0]],
                [point['CI95'][1] - point['point']]]), fmt='o', color=colors[i], capsize=4)
        rows = [r for r in traces if r['variant'] == v]
        steps = [r['step'] for r in rows[0]['trace']]
        if len(rows) != 18 or any([p['step'] for p in r['trace']] != steps for r in rows):
            raise ValueError('All18full cost traces per variant required')
        values = np.array([[p['loss'] for p in r['trace']] for r in rows])
        low, mid, high = np.quantile(values, [.1, .5, .9], axis=0)
        axes[2].plot(steps, mid, color=colors[i], label=v)
        axes[2].fill_between(steps, low, high, color=colors[i], alpha=.1)
    contrasts = [s['comparisons']['new_minus_old'], s['comparisons']['new_minus_physical'], s['new_head_neural_usage_increment']]
    for i, values in enumerate(contrasts):
        for shift, key, color in ((-.12, 'full_known_pp', '#227d99'), (.12, 'full_lower_pp', '#a85e36')):
            point = values[key]
            if point['CI95'] is not None:
                axes[1].errorbar(i + shift, point['point'], yerr=np.array([[point['point'] - point['CI95'][0]],
                    [point['CI95'][1] - point['point']]]), fmt='o', capsize=3, color=color,
                    label=('Known labels' if key == 'full_known_pp' else 'Completion lower') if i == 0 else None)
    axes[0].set_xticks(range(3), run.api.VARIANTS, rotation=18)
    axes[0].set_title('Full-policy known ADE utility')
    axes[0].set_ylabel('Gain vs frozen floor (%)')
    axes[1].set_xticks(range(3), ('new-old', 'new-physical', 'neural use'), rotation=18)
    axes[1].set_title('Trajectory-aware paired changes')
    axes[1].set_ylabel('Original floor-reference mass pp')
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].set_title('Actual cost-head training objective')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('Optimizer update')
    axes[2].set_ylabel('Minibatch median / p10-p90')
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis='y', color='#dddddd', lw=.6)
        ax.set_axisbelow(True)
    for ax in axes[:2]:
        ax.axhline(0, color='#888888', lw=.7)
    fig.suptitle('Matched frozen active banks: SOURCE fitting evidence only', y=.98)
    fig.text(.5, .025, '3,000 locality resamples; training spread is not a CI. No independent confirmation or deployment promotion.', ha='center', fontsize=8)
    fig.subplots_adjust(left=.06, right=.98, top=.84, bottom=.23, wspace=.38)
    folder = run.PUBLIC / 'figures'
    folder.mkdir(exist_ok=True)
    path = folder / 'summary.svg'
    fig.savefig(path, metadata={'Date': None})
    path.write_text('\n'.join(v.rstrip() for v in path.read_text().splitlines()) + '\n')
    fig.savefig(run.PRIVATE / 'summary.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('report', 'figures', 'bundle'))
    args = parser.parse_args()
    if args.phase == 'figures':
        figures()
    elif args.phase == 'bundle':
        print(json.dumps(bundle.pack(ROOT, run.PUBLIC)))
    else:
        value, summary = evidence()
        text, path = render(value, summary), run.PUBLIC / 'readout.md'
        if path.exists() and path.read_text() != text:
            raise ValueError('Frozen matched-bank report differs')
        path.write_text(text)
        print(json.dumps(summary['fitting_gates']))
