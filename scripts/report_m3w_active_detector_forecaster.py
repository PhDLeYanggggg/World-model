"""Artifacts for the actual active grouped/partial-neighbor producer experiment."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_active_detector_forecaster as run
from src.evaluation import m3w_metric_bundle as bundle


def number(value):
    return 'not_estimable' if value is None else f'{value:+.6f}'


def interval(value):
    if value['CI95'] is None:
        return 'not_estimable'
    return number(value['point']) + ' [' + ', '.join(number(v) for v in value['CI95']) + ']'


def support_profile(complete):
    seen = set()
    selected = []
    for ref in complete['groups']:
        path = ROOT / ref['path']
        if run.sha(path) != ref['sha256']:
            raise ValueError('Verified active readout group changed')
        row = json.loads(path.read_text())
        key = row['trial'], row['representation']
        if key in seen or row['kind'] != 'single':
            raise ValueError('Unique actual single-parent groups required')
        seen.add(key)
        if row['representation'] == 'history':
            selected.append(row['result'])
    if len(seen) != 36 or len(selected) != 9:
        raise ValueError('All36groups and9history producers required')
    records = set()
    sites = {}
    mean_gains, tail_gains = [], []
    mean_undefined = tail_undefined = 0
    for row in selected:
        for site, view in row['sites'].items():
            metrics = view['metrics']
            support = dict(indexed=metrics['all']['ADE']['indexed'], known_ADE=metrics['all']['ADE']['known'],
                           known_FDE=metrics['all']['FDE']['known'],
                           complete_future=metrics['complete_future']['ADE']['indexed'])
            if site in sites and sites[site] != support:
                raise ValueError('Matched active producers disagree on support')
            sites[site] = support
            for record_id, record in view['records'].items():
                records.add(record_id)
                ref, new = record['reference_ADE'], record['native_ADE']['history']
                if ref['mean'] is None or ref['mean'] <= 0:
                    mean_undefined += 1
                else:
                    mean_gains.append(100 * (1 - new['mean'] / ref['mean']))
                if ref['p95'] is None or ref['p95'] <= 0:
                    tail_undefined += 1
                else:
                    tail_gains.append(100 * (1 - new['p95'] / ref['p95']))
    return dict(producer_views=9, unique_recordings=len(records),
                indexed_row_views=sum(r['rows'] for r in selected),
                complete_future_row_views=sum(r['complete_future_rows'] for r in selected),
                partial_future_row_views=sum(r['partial_future_rows'] for r in selected),
                missing_future_row_views=sum(r['missing_future_rows'] for r in selected),
                stationary_observed_row_views=sum(r['stationary_observed_rows'] for r in selected),
                support_by_locality=sites,
                unique_queries=dict(indexed=sum(v['indexed'] for v in sites.values()),
                                    complete_future=sum(v['complete_future'] for v in sites.values()),
                                    partial_future=sum(v['known_ADE'] - v['complete_future'] for v in sites.values()),
                                    missing_future=sum(v['indexed'] - v['known_ADE'] for v in sites.values())),
                known_record_mean_tail_diagnostic=dict(defined_mean_views=len(mean_gains),
                    undefined_mean_views=mean_undefined, defined_p95_views=len(tail_gains),
                    undefined_p95_views=tail_undefined,
                    mean_worse_by_over2percent=sum(v < -2 for v in mean_gains),
                    p95_worse_by_over2percent=sum(v < -2 for v in tail_gains),
                    worst_mean_gain_percent=min(mean_gains) if mean_gains else None,
                    worst_p95_gain_percent=min(tail_gains) if tail_gains else None,
                    descriptive_not_approved_positive_harm_policy_gate=True),
                repeated_producer_seed_row_views_not_independent_samples=True)


def freeze_evidence(value):
    path = run.PUBLIC / 'evidence.json'
    if path.exists():
        previous = json.loads(path.read_text())
        # A later exact inference replay has new runtime metadata, not new science.
        if dict(value, replay=previous['replay']) != previous:
            raise ValueError('Frozen active scientific artifact differs')
        return previous
    run.once(path, value)
    return value


def evidence():
    check = json.loads((run.PUBLIC / 'verification.json').read_text())
    summary = json.loads((run.PUBLIC / 'summary.json').read_text())
    if (not check['all36groups_exact'] or not check['all9fresh_parent_predictions_exact']
            or check['summary_sha256'] != run.sha(run.PUBLIC / 'summary.json')
            or check['complete_sha256'] != run.sha(run.PUBLIC / 'complete.json')):
        raise ValueError('Separate complete exact active inference replay required')
    models = run.models()
    traces = []
    for (trial, arm), row in sorted(models.items()):
        loss = row['fit']['losses']
        if row['fit']['step'] != 4000 or loss[-1]['step'] != 4000 or row['fit']['held_rows_sampled']:
            raise ValueError('Full original training traces and no held-row draws required')
        traces.append(dict(trial=trial, seed=row['parent_trial']['seed'], representation=arm,
                           losses=loss, checkpoint_sha256=row['checkpoint']['sha256'],
                           minibatch_not_validation=True))
    run.once(run.PUBLIC / 'training_traces.json', dict(rows=traces, models=36,
        primary_optimizer_updates=144000, result_source='cached_verified_complete_active_training_traces',
        legacy_mean_past_normalized_ADE_trace_is_restored_pixel_ADE_before_loss_factors=True))
    frozen = json.loads((run.PUBLIC / 'training_freeze.json').read_text())
    complete = json.loads((run.PUBLIC / 'complete.json').read_text())
    losses = {}
    for arm in run.api.REPRESENTATIONS:
        rows = [r for r in traces if r['representation'] == arm]
        losses[arm] = dict(models=len(rows),
            first_median_minibatch_loss=float(np.median([r['losses'][0]['loss'] for r in rows])),
            final_median_minibatch_loss=float(np.median([r['losses'][-1]['loss'] for r in rows])))
    return freeze_evidence(dict(result_source='fresh_verified_actual_active_producer_aggregate_receipt',
        training={k: frozen[k] for k in ('pid', 'seconds', 'peak_RSS_bytes', 'count',
            'primary_optimizer_updates', 'extra_full_refit_updates', 'pilot_updates', 'disabled_exact_endpoints')},
        readout={k: complete[k] for k in ('pid', 'seconds', 'peak_RSS_bytes')},
        replay=json.loads((run.PRIVATE / 'verification_runtime.json').read_text()),
        checkpoint_bytes=sum(r['checkpoint']['bytes'] for r in models.values()),
        actual_active_backbone=True, prior_flat_backbone_study_not_a_matched_bank_replacement=True,
        loss_summary=losses, support_profile=support_profile(complete),
        physical_comparisons_preregistered=True,
        physical_contrasts_use_original_reference_mass_not_physical_denominator=True,
        artifact_code_bindings={str(p.relative_to(ROOT)): run.sha(p) for p in
            (Path(__file__), ROOT / 'tests/test_m3w_active_detector_artifacts.py')},
        summary_sha256=run.sha(run.PUBLIC / 'summary.json'),
        verification_sha256=run.sha(run.PUBLIC / 'verification.json'),
        proposal_gates=summary['proposal_input_gates'],
        proposal_input_evidence_gate=summary['proposal_input_evidence_gate'],
        learned_policy_safety_evaluated=False, controller_held_outcomes_opened=False,
        independent_roles_opened=False, deployment_changed=False, submission_ready=False,
        metric_claim=False, seconds_claim=False, stage5c_executed=False, smc_enabled=False)), summary


def render(value, summary):
    training, support = value['training'], value['support_profile']
    lines = ['# Detector History On The Actual Active Trajectory Backbone', '', '## Material Passport', '',
        'Fresh_run:36real Torch trajectory producers,144,000primary updates,',
        '4,000full-refit updates,900pilot updates,36fixed inference groups and a',
        'separate complete exact inference replay. Cached_verified: all9original',
        'active grouped/partial-neighbor dimensionless producers,318,969SOURCE',
        'rows and the verified163record past-only detector store. Not_run: a new',
        'learned gain/harm controller on this proposal bank,controller-held outcomes,',
        'independent selection/calibration/confirmation,DroneCrowd or deployment.', '',
        'This is the backbone actually consumed by the cost learner. The earlier',
        '72model flat complete-neighbor experiment remains a separate result,not',
        'a matched replacement of that active neural candidate. All9actual parent',
        'designs are retained;there is no active complement-parent family to drop.', '',
        f"Training PID{training['pid']}: {training['seconds']:.6f}s; peak RSS{training['peak_RSS_bytes']:,}B.",
        f"Readout PID{value['readout']['pid']}: {value['readout']['seconds']:.6f}s; separate replay PID{value['replay']['pid']}: {value['replay']['seconds']:.6f}s.",
        'All9complete disabled scientific states and fresh predictions reproduce',
        'their actual parents. First full history refit is exact. Native arm64CPU4,',
        'workers0,no resource probing,no NumPy training fallback or budget cut.', '',
        '## Fixed Matched Trajectory Accuracy', '',
        'Gain is relative to each producer own-fit-selected causal reference.',
        'Primary is history,not a winning seed or a post-readout ensemble. Three',
        'producer seeds and3,000locality resamples follow fold/seed averaging over',
        'all12required localities. All SOURCE readouts are development-exposed;',
        'locality exclusion and exact replay do not restore independent testing.', '',
        '| Detector input | Known ADE gain %, CI95 | FDE gain %, CI95 | Hard ADE gain %, CI95 | Easy ADE gain %, CI95 |',
        '|---|---|---|---|---|']
    for arm, metrics in summary['representations'].items():
        lines.append('| ' + arm + ' | ' + ' | '.join(interval(v) for v in
            (metrics['all']['ADE'], metrics['all']['FDE'], metrics['hard']['ADE'], metrics['easy']['ADE'])) + ' |')
    lines += ['', '## Support And Missing Outcomes', '',
        f"Unique queries:{support['unique_queries']['indexed']:,}; recordings:{support['unique_recordings']}; repeated producer/seed row views:{support['indexed_row_views']:,}.",
        f"Unique complete:{support['unique_queries']['complete_future']:,}; partial:{support['unique_queries']['partial_future']:,}; wholly unknown:{support['unique_queries']['missing_future']:,}.",
        'Counts are not multiplied by input arms. Repeated windows and producer',
        'views are not independent samples. Available-label ADE on a partial row',
        'does not reveal its full12step path. Locality082 has no requested final',
        'labels;undefined full-roster FDE/complete-future views are retained,not',
        'dropped or replaced by zero. A separate exact raw-label audit found no',
        'importer omission and no recoverable exact raw labels on these IDs.', '',
        '## History Minus Input Controls', '',
        '| Control | Known ADE change pp, CI95 | Completion-lower pp, CI95 | Completion-upper pp, CI95 |',
        '|---|---|---|---|']
    for arm, metrics in summary['history_contrasts']['all'].items():
        lines.append('| ' + arm + ' | ' + ' | '.join(interval(metrics[k]) for k in
            ('known_pp', 'lower_pp', 'upper_pp')) + ' |')
    lines += ['', 'Completion envelopes use causal forecast disagreement for wholly',
        'unknown outcomes on the original known reference-mass denominator.',
        'They are bounds,not observed losses or identified full-population gains.',
        'Hard/easy evaluation membership is observed-label diagnostic membership,',
        'not identified missing membership. No future mask or endpoint is an input.', '',
        '## All Six Preregistered Causal Physical Baselines', '',
        '| Fixed baseline | Known ADE gain %, CI95 | FDE gain %, CI95 | History-minus-fixed known ADE pp, CI95 | Completion-lower pp, CI95 |',
        '|---|---|---|---|---|']
    for key, metrics in summary['physical_baselines'].items():
        paired = summary['history_vs_fixed_physical']['all'][key]
        lines.append('| ' + run.source.BASELINES[int(key)] + ' | ' + ' | '.join(interval(v) for v in
            (metrics['ADE'], metrics['FDE'], paired['known_pp'], paired['lower_pp'])) + ' |')
    lines += ['', 'Physical contrasts were specified before scores. Differences use the',
        'original own-fit reference mass,not a new percent denominator for each',
        'physical method. No reference,model,threshold or alpha is chosen here.', '',
        '## Actual Training Loss', '',
        '| Input | Models | First median minibatch objective | Final median minibatch objective |',
        '|---|---:|---:|---:|']
    for arm, loss in value['loss_summary'].items():
        lines.append(f"| {arm} | {loss['models']} | {loss['first_median_minibatch_loss']:.6g} | {loss['final_median_minibatch_loss']:.6g} |")
    lines += ['', 'These are actual training minibatches,not validation or test losses.',
        'The legacy trace field mean_past_normalized_ADE contains restored pixel',
        'ADE before own-fit loss factors in this SOURCE adapter,not metres.',
        'The objective is differentiated through the active temporal/interaction',
        'Transformer;512new detector weights do not replace its original topology.', '',
        '## Proposal Evidence Gates', '',
        *['- ' + key + ': ' + json.dumps(v) for key, v in value['proposal_gates'].items()],
        '- proposal_input_evidence_gate: ' + json.dumps(value['proposal_input_evidence_gate']),
        '- learned_policy_safety_evaluated: false', '- deployment_changed: false',
        '- submission_ready: false', '',
        'No raw-proposal gate certifies the original2%complete all/easy per-required-',
        'recording policy-risk requirement. Empty selected risks stay not_estimable.',
        'A learned intervention policy must be trained and evaluated separately.', '',
        '## Scope And Limits', '',
        'SOURCE supplied detector-silver bbox midpoints,obs8/pred12/stride12 raw',
        'frames,pixel coordinates. Not human gold,verified live sensor-as-of,metres,',
        'seconds,true3D or a foundation model. The branch combines shape,confidence,',
        'body-relative displacement and overlap;it does not isolate a confidence-',
        'only mechanism. No new scene image,JEPA,generative rollout,Stage5C or SMC.',
        'No independent or89controller-held recording outcomes are opened. Local',
        'exact replay does not complete an anonymous from-scratch asset release.', '',
        '[Aggregate figure](figures/summary.svg), [actual loss traces](training_traces.json),',
        '[evidence](evidence.json), [model/data card](model_data_card.md),',
        '[failure analysis](failure_analysis.md), [Chinese reproduction guide](reproduction_zh.md).', '']
    return '\n'.join(lines)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    check = json.loads((run.PUBLIC / 'verification.json').read_text())
    if not check['all36groups_exact'] or check['summary_sha256'] != run.sha(run.PUBLIC / 'summary.json'):
        raise ValueError('Verified complete active results required')
    summary = json.loads((run.PUBLIC / 'summary.json').read_text())
    traces = json.loads((run.PUBLIC / 'training_traces.json').read_text())['rows']
    colors = ['#62696e', '#b84b37', '#856690', '#28749c']
    plt.rcParams.update({'font.size': 9, 'svg.hashsalt': run.NAME,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, arm in enumerate(run.api.REPRESENTATIONS):
        point = summary['representations'][arm]['all']['ADE']
        if point['CI95'] is not None:
            axes[0].errorbar(i, point['point'], yerr=np.array([[point['point'] - point['CI95'][0]],
                [point['CI95'][1] - point['point']]]), fmt='o', color=colors[i], capsize=4)
        rows = [r for r in traces if r['representation'] == arm]
        steps = [v['step'] for v in rows[0]['losses']]
        if len(rows) != 9 or any([v['step'] for v in r['losses']] != steps for r in rows):
            raise ValueError('All9full aligned actual loss traces per input required')
        values = np.array([[v['loss'] for v in r['losses']] for r in rows])
        low, mid, high = np.quantile(values, [.1, .5, .9], axis=0)
        axes[2].plot(steps, mid, color=colors[i], label=arm)
        axes[2].fill_between(steps, low, high, color=colors[i], alpha=.1)
    for i, arm in enumerate(('disabled', 'current', 'noise')):
        values = summary['history_contrasts']['all'][arm]
        for shift, bound, color in ((-.12, 'known_pp', '#28749c'), (.12, 'lower_pp', '#b84b37')):
            point = values[bound]
            if point['CI95'] is not None:
                axes[1].errorbar(i + shift, point['point'], yerr=np.array([[point['point'] - point['CI95'][0]],
                    [point['CI95'][1] - point['point']]]), fmt='o', color=color, capsize=3,
                    label=('Known labels' if bound == 'known_pp' else 'Completion lower bound') if i == 0 else None)
    axes[0].set_xticks(range(4), run.api.REPRESENTATIONS, rotation=20)
    axes[0].set_title('Actual active producer: known ADE')
    axes[0].set_ylabel('Gain vs own-fit causal reference (%)')
    axes[1].set_xticks(range(3), ('disabled', 'current', 'noise'), rotation=20)
    axes[1].set_title('History minus matched input controls')
    axes[1].set_ylabel('Change in original reference-mass pp')
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].set_yscale('log')
    axes[2].set_title('Real training objective, 9 models/input')
    axes[2].set_xlabel('Optimizer update')
    axes[2].set_ylabel('Minibatch median / p10-p90')
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes[:2]:
        ax.axhline(0, color='#808080', lw=.7)
    for ax in axes:
        ax.grid(axis='y', color='#dddddd', lw=.6)
        ax.set_axisbelow(True)
    fig.suptitle('Detector history on grouped / partial-neighbor dynamics: SOURCE development only', y=.98)
    fig.text(.5, .025, '3,000 locality resamples; loss spread is not a CI. No learned-policy safety or independent confirmation.', ha='center', fontsize=8)
    fig.subplots_adjust(left=.06, right=.98, top=.84, bottom=.22, wspace=.36)
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
        path, content = run.PUBLIC / 'readout.md', render(value, summary)
        if path.exists() and path.read_text() != content:
            raise ValueError('Frozen active report differs')
        path.write_text(content)
        print(json.dumps(value['proposal_gates']))
