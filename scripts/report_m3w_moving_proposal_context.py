"""Report a matched peer-context experiment without revising the risk contract."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_proposal_context as run
from scripts.report_m3w_moving_rollout_history import reduction, source_consistency

source, api, PUBLIC = run.source, run.api, run.PUBLIC


def comparisons(rows, jobs):
    expected = {(j['key'], arm) for j in jobs for arm in api.ARMS}
    actual = [(r['job'], r['arm']) for r in rows]
    if (len(jobs) != 3 or len(rows) != 6 or len(set(actual)) != 6 or set(actual) != expected
            or any(r['step'] != 1000 or r['parameters'] != 76548 for r in rows)):
        raise ValueError('All six unique full-budget matched models required')
    result = []
    for job in jobs:
        a, b = (next(r for r in rows if r['job'] == job['key'] and r['arm'] == arm) for arm in api.ARMS)
        result.append(dict(job=job['key'],
            mean_MSE_reduction_percent=reduction(a['final_fitting']['mean_MSE'], b['final_fitting']['mean_MSE']),
            **{k+'_reduction_percent':reduction(a['joint_attribution'][k], b['joint_attribution'][k])
                for k in ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')}))
    return result


def evidence_status(analysis, screen):
    held = analysis['status'] == 'fresh_run_exposed_development'
    if held != screen['followup_readout_allowed']:
        raise ValueError('Fixed fitting screen must control readout')
    if not held and ('held_policy' in analysis or 'contrasts' in analysis):
        raise ValueError('No held metrics permitted when readout was not run')
    return dict(held_readout='fresh_run_exposed_development' if held else 'not_run',
        held_reason=None if held else analysis['reason'],
        original_risk_pass=analysis['held_policy']['proposed_action_veto']['safety']['full_original_risk_pass']
            if held else 'not_run',
        independent_confirmation=False, deployment_changed=False, submission_ready=False)


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.hashsalt':'moving_proposal_context_v1', 'svg.fonttype':'none', 'font.size':9})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.subplots_adjust(left=.08, right=.985, bottom=.09, top=.84, hspace=.5, wspace=.42)
    jobs = sorted({r['job'] for r in rows}, key=lambda x:int(x.rsplit('critic', 1)[1]))
    colors = {'same_action':'#606060', 'proposed_action':'#1B8290'}
    for col, job in enumerate(jobs):
        arms = {r['arm']:r for r in rows if r['job'] == job}
        for arm in api.ARMS:
            t = arms[arm]['trace']
            axes[0, col].plot([r['step'] for r in t], [r['total'] for r in t], color=colors[arm], label=arm)
        axes[0, col].set(title=job.rsplit('_', 1)[1], xlabel='Continuation update', ylabel='Original joint batch loss')
        keys = ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')
        for i, arm in enumerate(api.ARMS):
            values = [arms[arm]['joint_attribution'][k]/arms['same_action']['joint_attribution'][k]
                if arms['same_action']['joint_attribution'][k] > 0 else float('nan') for k in keys]
            axes[1, col].bar(source.np.arange(3)+(i-.5)*.32, values, width=.32, color=colors[arm])
        axes[1, col].axhline(1, linestyle='--', color='#333333', linewidth=.7)
        axes[1, col].set(xticks=[0, 1, 2], xticklabels=['False safe', 'False reject', 'Joint cost'],
            ylabel='Fitting cost / same-action bridge')
        for row in (0, 1):
            axes[row, col].spines[['top', 'right']].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc='upper center', bbox_to_anchor=(.5, .94), frameon=False)
    fig.suptitle('Proposal-conditioned peer context: fitting, not safety evidence', y=.98, fontsize=11)
    out = PUBLIC/'figures'; out.mkdir(exist_ok=True)
    path = out/'context_fitting.svg'; fig.savefig(path, metadata={'Date':None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'context_fitting.png', dpi=150); plt.close(fig)


def main():
    analysis = source.read(PUBLIC/'analysis.json'); verify = source.read(PUBLIC/'verification.json')
    api.base.exact(verify['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    freeze = source.read(PUBLIC/'training_freeze.json')
    if (freeze['complete_heads'] != 6 or not freeze['matched_design_verified']
            or not source.read(PUBLIC/'full_refit.json')['exact'] or not verify['all6_fitting_metrics_and_attributions_exact']):
        raise ValueError('Six matched fits, full refit and metric replay required')
    rows = [source.read(source.checked(r)) for r in freeze['fits']]
    jobs = source.read(run.REG)['jobs']; matched = comparisons(rows, jobs)
    screen = api.screen(rows, jobs); api.base.exact(screen, analysis['fitting_screen'])
    status = evidence_status(analysis, screen); held = status['held_readout'] != 'not_run'
    phases = {k:source.read(PUBLIC/(k+'_execution.json')) for k in ('pilot', 'train', 'evaluate', 'verify')}
    summary = dict(status=analysis['status'], matched=matched, fitting_screen=screen, phases=phases, **status,
        context_audit=source.read(PUBLIC/'context_audit.json'), all6_real_fits=True, parameters_per_model=76548,
        primary_updates=6000, full_refit_updates=1000, pilot_updates=400, exact_resume=True,
        exact_full_refit=True, metric_replay_exact=True, parent_assets='cached_verified', models_and_analysis='fresh_run',
        held_policy=analysis.get('held_policy'), held_contrasts=analysis.get('contrasts'),
        source_consistency=source_consistency(analysis['contrasts']['main']) if held else None,
        original_rules_unchanged=True, identity_certification=False, original24group_primary='not_estimable_not_passed',
        no_new_H=True, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'result_summary.json', source.stable(summary)); figure(rows)
    f = lambda x:'NA' if x is None else f'{x:.4f}'
    text = ['# Proposal-Conditioned Peer Context', '',
        'I tested whether a neural risk head benefits from knowing which candidate',
        'each peer actually proposes. The matched control receives peers\' same-index',
        'hypothetical candidate. Both learn an equally sized, zero-initialized context',
        'bridge from the same completed parent, with the original loss and safety rules.', '',
        '## Matched Fitting Results', '',
        '| Context | Risk MSE reduction % | Joint-cost reduction % | False-safe-cost reduction % | False-reject-cost reduction % |',
        '|---|---:|---:|---:|---:|']
    for r in matched:
        text.append('| '+r['job'].rsplit('_', 1)[1]+' | '+' | '.join(f(r[k+'_reduction_percent']) for k in
            ('mean_MSE', 'total_joint_cost', 'false_safe_cost', 'false_reject_cost'))+' |')
    passed = sum(r['passed'] for r in screen['contexts'])
    text += ['', 'Positive reductions are better. These are own-fitting diagnostics, not',
        'held trajectory gains, independent confirmation or calibrated safety.',
        f'The fixed internal follow-up screen passes **{passed}/3** contexts.', '',
        'All six models complete1,000 updates with76,548 parameters. Initial predictions',
        'match the parent exactly. Both risk/decision encoders and both bridges update.',
        'Unknown-bound supervision, normalizers, draws and optimizer settings match.', '',
        '![Joint loss and decision costs](figures/context_fitting.svg)', '',
        '## Input Audit', '',
        '| Context | Rows | Complete queries | Mixed-proposal queries | Empty-proposal queries |',
        '|---|---:|---:|---:|---:|']
    for r in summary['context_audit']['rows']:
        text.append(f"| {r['job'].rsplit('_',1)[1]} | {r['rows']} | {r['queries']} | {r['multiple_action_queries']} | {r['empty_proposal_queries']} |")
    text += ['', 'All own-B proposals replay exactly from the original full legal-action mask',
        'and source-disjoint frozen proposer. A legal mask is never interpreted as an',
        'action set. Proposals precede the risk veto; this is not final admitted-set',
        'context or a physical interaction model. Overlapping queries are dependent.', '',
        '## Exposed Development', '']
    if held:
        text += ['| Policy | Known ADE gain % | Hard gain % | Switch % | Worst easy-net % | Original risk/support |',
            '|---|---:|---:|---:|---:|---:|']
        for name in ('physical_floor', 'constant_veto', 'same_action_veto', 'proposed_action_veto', 'proposed_action_risk_only_veto'):
            r = analysis['held_policy'][name]; s = r['safety']
            text.append('| '+name+' | '+' | '.join(f(x) for x in (r['known']['mean'], r['hard']['mean'],
                r['switch_percent']['mean'], s['worst_defined_signed_easy_net_percent']))+f" | {s['supported']}/{s['required']} |")
        c = analysis['contrasts']['main']
        text += ['', f"Proposed-minus-same contrast: {f(c['known_pp']['mean'])} pp; missing-label completion bounds "
            f"[{f(c['completion_lower_pp']['mean'])}, {f(c['completion_upper_pp']['mean'])}] pp, not sampling CIs.",
            'All controls and source contrasts remain in analysis.json. Already exposed',
            'nested-C is development, not a new independent test. No model is promoted.']
    else:
        text += ['**not_run:** '+analysis['reason']+'.',
            'No additional development/test gain is inferred. The previous17-view',
            'risk/support failure is unchanged. Lower fitting error is not safety.']
    text += ['', '## Execution And Scope', '', '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for key, r in phases.items():
        text.append(f"| {key} | {r['pid']} | {r['seconds']:.2f} | {r['peak_RSS_bytes']} |")
    text += ['', '6,000 primary +1,000 exact-refit +400 pilot updates. Native arm64 CPU4,',
        'inter-op1/workers0; checkpoints and heartbeat. Three contexts share one',
        'producer seed and do not constitute a three-seed independent confirmation.', '',
        '- fresh_run: six neural fits, causal-context audit and exact metric replay.',
        '- cached_verified: parent weights, data/schema, source roles and normalizers.',
        '- not_run: reserved/H confirmation, new trajectory training, JEPA or deployment.',
        '- Original2% all/easy positive-harm, easy-net and reference/support rules retained.',
        '- Empty support stays NA. No identity certificate or threshold search.',
        '- Original24-group primary stays NA. Moving20, DUT, DroneCrowd and H stay closed.',
        '- Native obs8/pred12 image coordinates, not metric/seconds/true3D/foundation/gold.',
        '- Stage5C/SMC disabled. The paper is not yet submission-ready.', '',
        '[Protocol](protocol.md) | [Metrics](result_summary.json) | [Failure analysis](failure_analysis.md)',
        '| [Reproduction](operation_zh.md)', '']
    (PUBLIC/'README.md').write_text('\n'.join(text))
    (PUBLIC/'gates.md').write_text('\n'.join(['# Evidence Gates', '', '| Gate | Status |', '|---|---|',
        '| Six real fits / exact resume / full refit | pass |', '| Initial identity / causal proposal replay | pass |',
        '| All fitting metrics and attributions replay | pass |', f'| Internal fitting screen | {passed}/3 |',
        f"| Original held risk/support | {status['original_risk_pass']} |", '| Original24-group primary | NA, not pass |',
        '| Independent confirmation | not_run |', '| Deployment / submission readiness | false / false |',
        '| Stage5C / SMC | disabled / disabled |', '', 'Engineering checks are not scientific safety passes.', '']))
    print(source.json.dumps(dict(fits=6, fitting_screen=passed, **status)))


if __name__ == '__main__':
    main()
