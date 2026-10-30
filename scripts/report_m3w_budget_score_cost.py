"""Verified budget-score artifacts, complete actual traces and compact metrics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_budget_score_cost as run
from scripts import report_m3w_active_bank_cost as prior
from src.evaluation import m3w_metric_bundle as bundle

interval = prior.interval


def support_profile(complete):
    grouped = {o: [] for o in run.api.OBJECTIVES}
    keys = set()
    for ref in complete['groups']:
        path = ROOT / ref['path']
        if run.sha(path) != ref['sha256']:
            raise ValueError('Frozen budget-score aggregate changed')
        row = json.loads(path.read_text())
        key = row['context'], row['source'], row['seed'], row['objective_arm']
        if key in keys or row['objective_arm'] not in grouped:
            raise ValueError('Repeated or unregistered objective group')
        keys.add(key)
        grouped[row['objective_arm']].append(ref)
    if len(keys) != 216:
        raise ValueError('All216objective groups required')
    profiles = {o: prior.support_profile(dict(groups=refs)) for o, refs in grouped.items()}
    reference = profiles['original']
    if any(p != reference for p in profiles.values()):
        raise ValueError('Objective arms must have identical query support')
    return dict(reference, repeated_row_views_per_objective=reference['repeated_row_views'],
        repeated_row_views_all_objectives=sum(p['repeated_row_views'] for p in profiles.values()),
        objectives=3, objective_banks_and_seeds_not_independent_samples=True)


def freeze(value):
    path = run.PUBLIC / 'evidence.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if dict(value, replay=previous['replay']) != previous:
            raise ValueError('Frozen score evidence changed')
        return previous
    run.once(path, value)
    return value


def evidence():
    check = json.loads((run.PUBLIC / 'verification.json').read_text())
    if (not check['all216groups_exact'] or not check['all72original_predictions_actions_full_scores_exact']
            or check['summary_sha256'] != run.sha(run.PUBLIC / 'summary.json')
            or check['complete_sha256'] != run.sha(run.PUBLIC / 'complete.json')):
        raise ValueError('Separate complete216group exact replay required')
    models, traces = run.models(), []
    for (context, seed, variant, objective), row in sorted(models.items()):
        state = run.frozen.api.completion.base.read_checkpoint(ROOT / row['checkpoint']['path'])
        if state['step'] != 2000 or state['trace'][-1]['step'] != 2000 or state['unknown_rows_with_loss']:
            raise ValueError('All162actual complete known-only training traces required')
        traces.append(dict(context=context, seed=seed, variant=variant, objective_arm=objective,
            actual_objective=state['objective'], trace=state['trace'], checkpoint_sha256=row['checkpoint']['sha256'],
            minibatch_not_validation=True))
    run.once(run.PUBLIC / 'training_traces.json', dict(rows=traces, models=162,
        primary_optimizer_updates=324000, result_source='cached_verified_actual_Torch_training_traces'))
    loss = {}
    for objective in run.api.OBJECTIVES:
        loss[objective] = {}
        for variant in run.api.active.VARIANTS:
            selected = [r for r in traces if r['objective_arm'] == objective and r['variant'] == variant]
            if len(selected) != 18:
                raise ValueError('All18matched models per objective/bank required')
            loss[objective][variant] = dict(models=18,
                first_median=float(np.median([r['trace'][0]['loss'] for r in selected])),
                final_median=float(np.median([r['trace'][-1]['loss'] for r in selected])))
    complete = json.loads((run.PUBLIC / 'complete.json').read_text())
    training = json.loads((run.PUBLIC / 'training_freeze.json').read_text())
    summary = json.loads((run.PUBLIC / 'summary.json').read_text())
    value = dict(result_source='verified_real_budget_score_cost_training_and_complete_policy_replay',
        training={k: training[k] for k in ('pid', 'seconds', 'peak_RSS_bytes', 'count',
            'primary_optimizer_updates', 'extra_full_refit_updates', 'pilot_updates', 'exact_original_controls')},
        readout={k: complete[k] for k in ('pid', 'seconds', 'peak_RSS_bytes')},
        replay=json.loads((run.PRIVATE / 'verification_runtime.json').read_text()),
        checkpoint_bytes=sum(r['checkpoint']['bytes'] for r in models.values()), loss_summary=loss,
        support_profile=support_profile(complete), summary_sha256=run.sha(run.PUBLIC / 'summary.json'),
        verification_sha256=run.sha(run.PUBLIC / 'verification.json'),
        artifact_code_bindings={str(p.relative_to(ROOT)): run.sha(p) for p in
            (Path(__file__), ROOT / 'tests/test_m3w_budget_score_artifacts.py')},
        fitting_gates=summary['fitting_gates'], fitting_advancement_gate=summary['fitting_advancement_gate'],
        independent_confirmation=False, deployment_changed=False, submission_ready=False,
        held_scored=False, independent_roles_opened=False, metric_claim=False, seconds_claim=False,
        stage5c_executed=False, smc_enabled=False)
    return freeze(value), summary


def render(e, s):
    t, p = e['training'], e['support_profile']
    out = ['# Original-Budget Score Learning: Complete Matched Result', '',
        '## Material Passport', '',
        'Fresh_run:162real Torch cost heads,324,000primary updates,2,000extra',
        'full-refit updates,420real pilot updates,216fixed readouts and separate',
        'complete exact policy inference replay. Cached_verified:previous SOURCE',
        'inputs,original controls and frozen old/new active-producer banks.',
        'Not_run:controller-held/independent outcomes,new trajectory producer,',
        'scene-joint decision training or deployment.', '',
        f"Training PID{t['pid']}:{t['seconds']:.6f}s,peak RSS{t['peak_RSS_bytes']:,}B.",
        f"Initial readout PID{e['readout']['pid']}:{e['readout']['seconds']:.6f}s; separate replay PID{e['replay']['pid']}:{e['replay']['seconds']:.6f}s.",
        'All54original full scientific states and all72original prediction/action/',
        'full result controls are exact. The first upper_score new-bank model',
        'has an exact independent2,000step refit. Native arm64 CPU4/workers0,',
        'no NumPy training fallback and no shortened model or update roster.', '',
        '## Controlled Learning Change', '',
        'Keep identified five-moment regression and within-row ranking. Add',
        'signed H-0.02R and H_easy-0.02R_easy residual supervision,once with',
        'symmetric squared error and once with4xunderprediction weight.',
        'The original risk rule,decoder,banks,sampler,features and normalizers',
        'are unchanged. No supervised score,target or future mask enters',
        'inference. These are estimated scores,not a calibrated risk theorem.', '',
        f"Unique queries:{p['unique_queries']:,}; recordings:{p['unique_recordings']}; localities:12.",
        f"Known:{p['unique_known_queries']:,}; wholly unknown:{p['unique_unknown_queries']:,}.",
        f"Repeated row views per objective:{p['repeated_row_views_per_objective']:,}; all objectives:{p['repeated_row_views_all_objectives']:,}.",
        'Counts across contexts,seeds,objectives and banks are not independent',
        'samples. Intervals use3,000locality resamples after context/head-seed',
        'averaging. SOURCE has historical development exposure; this is fitting',
        'resubstitution,not independent cross-scene confirmation.', '',
        '## Complete-Policy Accuracy And Recording Risk', '',
        '| Objective | Bank | Known ADE gain %, CI95 | Required recording risk support | Known easy violations | Undefined easy views | Switch row views |',
        '|---|---|---|---:|---:|---:|---:|']
    for objective, values in s['objectives'].items():
        for variant in run.api.active.VARIANTS:
            risk = values['safety'][variant]
            out.append(f"| {objective} | {variant} | {interval(values['accuracy'][variant])} | {risk['supported_record_views']}/{risk['required_record_views']} | {risk['known_easy_risk_violations']} | {risk['undefined_easy_record_views']} | {risk['selected']} |")
    out += ['', 'Safety keeps the complete unchanged all/easy2%rule on every required',
        'recording,including missing-outcome bounds. Pooled easy net gain is not',
        'the selected-easy positive-harm risk. Full-roster FDE remains undefined',
        'when a required locality lacks requested final labels.', '',
        '## Paired Objective Effects On The Same Bank', '',
        '| Objective | Bank | Full known pp, CI95 | Completion lower pp, CI95 | Same-count lower pp, CI95 |',
        '|---|---|---|---|---|']
    for objective in ('symmetric_score', 'upper_score'):
        for variant in run.api.active.VARIANTS:
            d = s['objectives'][objective]['against_original'][variant]
            out.append(f"| {objective} | {variant} | {interval(d['full_known_pp'])} | {interval(d['full_lower_pp'])} | {interval(d['matched_lower_pp'])} |")
    main = s['objectives']['upper_score']
    out += ['', 'Primary remains upper_score/new_neural minus original/new_neural,',
        'not the best observed arm. Same-count controls measure utility only;',
        'empty comparisons are not_estimable,not safe zero. All contrasts use',
        'actual selected trajectories and the original known floor-reference mass.', '',
        '- Upper new versus original old,known pp: ' + interval(main['against_original']['new_minus_original_old']['full_known_pp']),
        '- Upper new versus original old,completion lower pp: ' + interval(main['against_original']['new_minus_original_old']['full_lower_pp']),
        '- Upper new versus symmetric new,known pp: ' + interval(main['against_symmetric']['new_neural']['full_known_pp']),
        '- Same upper new head,actual neural-use known pp: ' + interval(main['actual_neural_use']['full_known_pp']),
        '- Same upper new head,actual neural-use completion lower pp: ' + interval(main['actual_neural_use']['full_lower_pp']), '',
        'The same-head neural restriction is not another fit. It separates',
        'actual neural path use from shared-head physical-choice changes.', '',
        '## Actual Training Objective', '',
        '| Objective | Bank | Heads | First median minibatch loss | Final median minibatch loss |',
        '|---|---|---:|---:|---:|']
    for objective, values in e['loss_summary'].items():
        for variant, loss in values.items():
            out.append(f"| {objective} | {variant} | {loss['models']} | {loss['first_median']:.6g} | {loss['final_median']:.6g} |")
    out += ['', 'Objective definitions differ,so loss magnitude is not a comparable',
        'accuracy ranking across arms. Curves are real fitting minibatches,not',
        'validation losses. Unknown rows receive no observed-cost supervision.', '',
        '## Fixed Fitting-Progress Gates', '',
        *['- ' + k + ': ' + json.dumps(v) for k, v in s['fitting_gates'].items()],
        '- fitting_advancement_gate: ' + json.dumps(s['fitting_advancement_gate']),
        '- independent_confirmation: false', '- deployment_changed: false', '- submission_ready: false', '',
        'SOURCE remains supplied detector-silver bbox-midpoint pixels,obs8/pred12/',
        'stride12raw frames. Not historical SDDt50,human gold,verified live',
        'sensor-as-of,metric/seconds,true3D or foundation evidence. No Stage5C',
        'execution orSMC. Independent evidence and scene-joint decision gaps remain.', '',
        '[Actual curves](figures/summary.svg),[machine evidence](evidence.json),',
        '[failure analysis](failure_analysis.md),[model/data card](model_data_card.md),',
        '[paper addendum](paper_addendum.md),[Chinese guide](reproduction_zh.md).', '']
    return '\n'.join(out)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = json.loads((run.PUBLIC / 'summary.json').read_text())
    e = json.loads((run.PUBLIC / 'evidence.json').read_text())
    if e['summary_sha256'] != run.sha(run.PUBLIC / 'summary.json'):
        raise ValueError('Verified score summary required')
    rows = json.loads((run.PUBLIC / 'training_traces.json').read_text())['rows']
    plt.rcParams.update({'font.size': 9, 'svg.hashsalt': run.NAME, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = dict(original='#666666', symmetric_score='#ae6437', upper_score='#227d99')
    banks = list(run.api.active.VARIANTS)
    for i, objective in enumerate(run.api.OBJECTIVES):
        values = s['objectives'][objective]
        for j, variant in enumerate(banks):
            point = values['accuracy'][variant]
            if point['CI95'] is not None:
                axes[0, 0].errorbar(j + (i - 1) * .17, point['point'],
                    yerr=np.array([[point['point'] - point['CI95'][0]], [point['CI95'][1] - point['point']]]),
                    fmt='o', color=colors[objective], capsize=3, label=objective if j == 0 else None)
            risk = values['safety'][variant]
            axes[1, 0].bar(j + (i - 1) * .22, risk['supported_record_views'], width=.2,
                color=colors[objective], label=objective if j == 0 else None)
        trace_rows = [r for r in rows if r['objective_arm'] == objective and r['variant'] == 'new_neural']
        steps = [p['step'] for p in trace_rows[0]['trace']]
        if len(trace_rows) != 18 or any([p['step'] for p in r['trace']] != steps for r in trace_rows):
            raise ValueError('All18complete new-bank traces per objective required')
        loss = np.array([[p['loss'] for p in r['trace']] for r in trace_rows])
        low, median, high = np.quantile(loss, [.1, .5, .9], axis=0)
        axes[1, 1].plot(steps, median, color=colors[objective], label=objective)
        axes[1, 1].fill_between(steps, low, high, color=colors[objective], alpha=.1)
    for j, variant in enumerate(banks):
        values = s['objectives']['upper_score']['against_original'][variant]
        for shift, key, color in ((-.1, 'full_known_pp', '#227d99'), (.1, 'full_lower_pp', '#ae6437')):
            point = values[key]
            if point['CI95'] is not None:
                axes[0, 1].errorbar(j + shift, point['point'],
                    yerr=np.array([[point['point'] - point['CI95'][0]], [point['CI95'][1] - point['point']]]),
                    fmt='o', capsize=3, color=color,
                    label=('Known labels' if key == 'full_known_pp' else 'Completion lower') if j == 0 else None)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks(range(3), banks)
    axes[0, 0].set_title('Complete-policy known ADE utility')
    axes[0, 0].set_ylabel('Gain vs frozen floor (%)')
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set_title('Upper-score minus original objective')
    axes[0, 1].set_ylabel('Original reference mass pp')
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].set_title('Recording-risk support: all 444 views required')
    axes[1, 0].set_ylabel('Supported recording views')
    axes[1, 0].axhline(444, color='#888888', ls='--', lw=.7)
    axes[1, 0].set_ylim(0, 465)
    axes[1, 1].set_title('Actual new-bank training objectives')
    axes[1, 1].set_xlabel('Optimizer update')
    axes[1, 1].set_ylabel('Minibatch median / p10-p90, log scale')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', color='#dddddd', lw=.6)
        ax.set_axisbelow(True)
    for ax in axes[0]:
        ax.axhline(0, color='#888888', lw=.7)
    fig.suptitle('Original-budget score learning: SOURCE fitting evidence only', y=.98)
    fig.text(.5, .025, '3,000 locality resamples; loss definitions differ and training spread is not a CI. No independent or deployment promotion.', ha='center', fontsize=8)
    fig.subplots_adjust(left=.07, right=.98, top=.92, bottom=.10, hspace=.38, wspace=.3)
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
    if args.phase == 'bundle':
        print(json.dumps(bundle.pack(ROOT, run.PUBLIC)))
    elif args.phase == 'figures':
        figures()
    else:
        value, summary = evidence()
        path, text = run.PUBLIC / 'readout.md', render(value, summary)
        if path.exists() and path.read_text() != text:
            raise ValueError('Frozen budget-score report differs')
        path.write_text(text)
        print(json.dumps(summary['fitting_gates']))
