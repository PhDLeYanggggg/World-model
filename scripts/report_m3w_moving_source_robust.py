"""Report all matched source-loss fits without changing the risk contract."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_source_robust as run
from scripts.report_m3w_moving_rollout_history import reduction, source_consistency

source, api, PUBLIC = run.source, run.api, run.PUBLIC


def comparisons(rows, jobs):
    expected = {(j['key'], arm) for j in jobs for arm in api.ARMS}
    actual = [(r['job'], r['arm']) for r in rows]
    if (len(jobs) != 3 or len(rows) != 6 or len(set(actual)) != 6 or set(actual) != expected
            or any(r['step'] != 1000 or r['parameters'] != 68356 for r in rows)):
        raise ValueError('All six unique full-budget models required')
    result = []
    for job in jobs:
        a, b = (next(r for r in rows if r['job'] == job['key'] and r['arm'] == arm) for arm in api.ARMS)
        per_source = []
        for site, x in a['final_sources']['by_source'].items():
            y = b['final_sources']['by_source'][site]
            per_source.append(dict(source=site, queries=x['queries'],
                objective_reduction_percent=reduction(x['total'], y['total']),
                **{k+'_reduction_percent': reduction(x['joint'][k], y['joint'][k])
                   for k in ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')}))
        result.append(dict(job=job['key'],
            mean_MSE_reduction_percent=reduction(a['final_fitting']['mean_MSE'], b['final_fitting']['mean_MSE']),
            worst_source_objective_reduction_percent=reduction(
                a['final_sources']['worst_source_objective'], b['final_sources']['worst_source_objective']),
            **{k+'_reduction_percent': reduction(a['joint_attribution'][k], b['joint_attribution'][k])
               for k in ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')},
            per_source=per_source))
    return result


def evidence_status(analysis, screen):
    held = analysis['status'] == 'fresh_run_exposed_development'
    if held != screen['followup_readout_allowed']:
        raise ValueError('Fixed fitting screen must control readout')
    if not held and ('held_policy' in analysis or 'contrasts' in analysis):
        raise ValueError('No held metrics permitted when readout was not run')
    return dict(held_readout='fresh_run_exposed_development' if held else 'not_run',
        held_reason=None if held else analysis['reason'],
        original_risk_pass=analysis['held_policy']['source_robust_veto']['safety']['full_original_risk_pass']
            if held else 'not_run',
        independent_confirmation=False, deployment_changed=False, submission_ready=False)


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.hashsalt': 'moving_source_robust_v1', 'svg.fonttype': 'none', 'font.size': 9})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.subplots_adjust(left=.08, right=.985, bottom=.09, top=.84, hspace=.5, wspace=.42)
    jobs = sorted({r['job'] for r in rows}, key=lambda x: int(x.rsplit('critic', 1)[1]))
    colors = {'mean_continue': '#606060', 'source_robust': '#1B8290'}
    for col, job in enumerate(jobs):
        arms = {r['arm']: r for r in rows if r['job'] == job}
        for arm in api.ARMS:
            trace = arms[arm]['trace']
            axes[0, col].plot([r['step'] for r in trace], [r['original_total'] for r in trace],
                color=colors[arm], label=arm, linewidth=1.1)
        axes[0, col].set(title=job.rsplit('_', 1)[1], xlabel='Continuation update', ylabel='Common original batch loss')
        keys = ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')
        for i, arm in enumerate(api.ARMS):
            values = [arms[arm]['joint_attribution'][k]/arms['mean_continue']['joint_attribution'][k]
                if arms['mean_continue']['joint_attribution'][k] > 0 else float('nan') for k in keys]
            axes[1, col].bar(source.np.arange(3)+(i-.5)*.32, values, width=.32, color=colors[arm])
        axes[1, col].axhline(1, linestyle='--', color='#333333', linewidth=.7)
        axes[1, col].set(xticks=[0, 1, 2], xticklabels=['False safe', 'False reject', 'Joint cost'],
            ylabel='Fitting cost / mean continuation')
        for row in (0, 1):
            axes[row, col].spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(.5, .94), frameon=False)
    fig.suptitle('Source-loss continuation: fitting diagnostics, not safety evidence', y=.98, fontsize=11)
    out = PUBLIC/'figures'; out.mkdir(exist_ok=True)
    path = out/'source_fitting.svg'; fig.savefig(path, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'source_fitting.png', dpi=150); plt.close(fig)


def main():
    analysis = source.read(PUBLIC/'analysis.json'); verify = source.read(PUBLIC/'verification.json')
    api.base.exact(verify['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    freeze = source.read(PUBLIC/'training_freeze.json')
    if (freeze['complete_heads'] != 6 or not freeze['matched_design_verified']
            or not source.read(PUBLIC/'full_refit.json')['exact']
            or not verify['all6_fitting_source_metrics_and_attributions_exact']):
        raise ValueError('Six matched fits, full refit and metric replay required')
    rows = [source.read(source.checked(r)) for r in freeze['fits']]
    jobs = source.read(run.REG)['jobs']; matched = comparisons(rows, jobs)
    screen = api.screen(rows, jobs); api.base.exact(screen, analysis['fitting_screen'])
    status = evidence_status(analysis, screen); held = status['held_readout'] != 'not_run'
    phases = {k:source.read(PUBLIC/(k+'_execution.json')) for k in ('pilot', 'train', 'evaluate', 'verify')}
    summary = dict(status=analysis['status'], matched=matched, fitting_screen=screen, phases=phases,
        **status, all6_real_fits=True, parameters_per_model=68356, primary_updates=6000,
        full_refit_updates=1000, pilot_updates=400, exact_resume=True, exact_full_refit=True,
        metric_replay_exact=True, parent_assets='cached_verified', models_and_analysis='fresh_run',
        held_policy=analysis.get('held_policy'), held_contrasts=analysis.get('contrasts'),
        source_consistency=source_consistency(analysis['contrasts']['main']) if held else None,
        original_rules_unchanged=True, identity_certification=False,
        original24group_primary='not_estimable_not_passed', no_new_H=True,
        stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'result_summary.json', source.stable(summary)); figure(rows)
    f = lambda x: 'NA' if x is None else f'{x:.4f}'
    text = ['# Source-Robust Neural Continuation', '',
        'I tested whether giving the worst fitting source a fixed extra loss weight',
        'improves intervention-risk learning. Six actual neural fits continue the',
        'same parent models, with identical inputs, initialization, draws and budget.',
        'This changes training, not the original evaluation contract.', '',
        '## Matched Fitting Contrast', '',
        '| Context | Worst-source objective reduction % | Risk MSE reduction % | Joint-cost reduction % | False-safe-cost reduction % | False-reject-cost reduction % |',
        '|---|---:|---:|---:|---:|---:|']
    for r in matched:
        text.append('| '+r['job'].rsplit('_', 1)[1]+' | '+' | '.join(f(r[k+'_reduction_percent']) for k in
            ('worst_source_objective', 'mean_MSE', 'total_joint_cost', 'false_safe_cost', 'false_reject_cost'))+' |')
    passed = sum(r['passed'] for r in screen['contexts'])
    text += ['', 'Positive reductions are better. These are own-fitting results, not',
        'held trajectory improvements, independent confirmation or calibrated safety.',
        f'The fixed internal follow-up screen passes **{passed}/3** contexts.', '',
        'The treatment is 0.75 original loss + 0.25 worst-source loss. The control',
        'continues the original mean loss. Each model has 68,356 parameters, 524',
        'input fields and 1,000 additional updates after its 2,000-update parent.',
        'Unknown targets retain their original bounds. Source names group training',
        'loss only, not inference inputs. Both encoders and all parameters update.', '',
        '![Common loss and decision costs](figures/source_fitting.svg)', '',
        'The plot compares the same original loss on matched sampled batches, not',
        'two differently weighted objective values. Full-data costs are separate.', '',
        '## Exposed Development', '']
    if held:
        text += ['| Policy | Known ADE gain % | Hard gain % | Switch % | Worst easy-net % | Original risk/support |',
            '|---|---:|---:|---:|---:|---:|']
        for name in ('physical_floor', 'constant_veto', 'mean_continue_veto', 'source_robust_veto', 'source_robust_risk_only_veto'):
            r = analysis['held_policy'][name]; s = r['safety']
            text.append('| '+name+' | '+' | '.join(f(x) for x in (r['known']['mean'], r['hard']['mean'],
                r['switch_percent']['mean'], s['worst_defined_signed_easy_net_percent']))+
                f" | {s['supported']}/{s['required']} |")
        c = analysis['contrasts']['main']
        text += ['', f"Registered robust-minus-mean contrast: {f(c['known_pp']['mean'])} pp; "
            f"missing-label completion bounds [{f(c['completion_lower_pp']['mean'])}, "
            f"{f(c['completion_upper_pp']['mean'])}] pp. These are not sampling CIs.",
            'All controls, source contrasts and completion bounds remain in analysis.json.',
            'This exposed nested-C evaluation is development, not a fresh independent test.']
    else:
        text += ['**not_run:** '+analysis['reason']+'.',
            'No additional development or test improvement is inferred. The previous',
            '17-view risk/support failure remains unchanged, not converted into a pass.']
    text += ['', '## Execution', '', '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for key, r in phases.items():
        text.append(f"| {key} | {r['pid']} | {r['seconds']:.2f} | {r['peak_RSS_bytes']} |")
    text += ['', '6,000 primary + 1,000 exact full-refit + 400 real pilot updates. Native arm64',
        'PyTorch, CPU4/inter-op1/workers0; checkpoints and heartbeat. Three contexts',
        'share one producer seed and are not three independent training seeds.', '',
        '- `fresh_run`: six continuations, source losses, error attribution and exact replay.',
        '- `cached_verified`: parent weights, causal data, schema, own-B source partitions.',
        '- `not_run`: reserved/H confirmation, a new trajectory predictor, JEPA and deployment.',
        '- Original 2% all/easy positive-harm, easy-net and reference/support rules retained.',
        '- Empty support remains NA; no identity certification or threshold search.',
        '- Original 24-group primary stays NA. Moving20, DUT, DroneCrowd and H stay closed.',
        '- Native obs8/pred12 image coordinates; no metric/seconds/true3D/foundation claim.',
        '- Stage5C/SMC remain off; the paper is not yet submission-ready.', '',
        '[Protocol](protocol.md) | [Source losses](source_losses.md) | [Metrics](result_summary.json)',
        '| [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)', '']
    (PUBLIC/'README.md').write_text('\n'.join(text))
    text = ['# Per-Source Fitting Costs', '', 'Matched robust-minus-mean continuation; reductions are positive when better.',
        'Sources below are fitting sources, not independent held-out tests.', '',
        '| Context | Source | Queries | Objective reduction % | Joint-cost reduction % | False-safe-cost reduction % | False-reject-cost reduction % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in matched:
        for s in r['per_source']:
            text.append('| '+r['job'].rsplit('_', 1)[1]+' | '+s['source']+' | '+str(s['queries'])+' | '+
                ' | '.join(f(s[k+'_reduction_percent']) for k in ('objective', 'total_joint_cost', 'false_safe_cost', 'false_reject_cost'))+' |')
    text += ['', 'A zero control denominator produces NA, never a claimed perfect reduction.',
        'Query counts describe the sampler units, not independent real-world scenes.', '']
    (PUBLIC/'source_losses.md').write_text('\n'.join(text))
    (PUBLIC/'gates.md').write_text('\n'.join(['# Evidence Gates', '', '| Gate | Status |', '|---|---|',
        '| Six matched real fits / resume / full refit | pass |',
        '| All fitting metrics, source losses and attributions replay | pass |',
        f'| Internal fitting screen | {passed}/3 |', f"| Original held risk/support | {status['original_risk_pass']} |",
        '| Original 24-group primary | NA, not passed |', '| Independent confirmation | not_run |',
        '| Deployment change / submission readiness | false / false |', '| Stage5C / SMC | disabled / disabled |', '',
        'Engineering passes are not counted as scientific safety passes.', '']))
    print(source.json.dumps(dict(fits=6, fitting_screen=passed, **status)))


if __name__ == '__main__':
    main()
