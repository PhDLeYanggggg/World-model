"""Summarize every preregistered model, without promoting a held winner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_rollout_history as run

source, api, PUBLIC = run.source, run.api, run.PUBLIC


def reduction(a, b):
    return None if a == 0 else 100*(a-b)/a


def comparisons(rows, jobs):
    expected = {(j['key'], arm) for j in jobs for arm in api.ARMS}
    actual = [(r['job'], r['arm']) for r in rows]
    if (len(jobs) != 3 or len(rows) != 6 or len(set(actual)) != 6 or set(actual) != expected
            or any(r['step'] != 2000 or r['parameters'] != 68356 for r in rows)):
        raise ValueError('All six unique full-budget models required')
    result = []
    for job in jobs:
        a, b = (next(r for r in rows if r['job'] == job['key'] and r['arm'] == arm) for arm in api.ARMS)
        result.append(dict(job=job['key'],
            mean_MSE_reduction_percent=reduction(a['final_fitting']['mean_MSE'], b['final_fitting']['mean_MSE']),
            **{k+'_reduction_percent': reduction(a['joint_attribution'][k], b['joint_attribution'][k])
               for k in ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')}))
    return result


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.hashsalt': 'moving_rollout_history_v1', 'svg.fonttype': 'none', 'font.size': 9})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.subplots_adjust(left=.075, right=.985, bottom=.09, top=.84, hspace=.5, wspace=.4)
    jobs = sorted({r['job'] for r in rows}, key=lambda x: int(x.rsplit('critic', 1)[1]))
    colors = {'masked': '#606060', 'relational': '#1B8290'}
    for column, job in enumerate(jobs):
        arms = {r['arm']: r for r in rows if r['job'] == job}
        for arm in api.ARMS:
            trace = arms[arm]['trace']
            axes[0, column].plot([r['step'] for r in trace], [r['risk_MSE'] for r in trace],
                color=colors[arm], label=arm, linewidth=1.1)
        axes[0, column].set(title=job.rsplit('_', 1)[1], xlabel='Optimizer update', ylabel='Sampled-batch risk MSE')
        keys = ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')
        for i, arm in enumerate(api.ARMS):
            values = [arms[arm]['joint_attribution'][k]/arms['masked']['joint_attribution'][k] for k in keys]
            axes[1, column].bar(source.np.arange(3)+(i-.5)*.32, values, width=.32, color=colors[arm])
        axes[1, column].axhline(1, linestyle='--', color='#333333', linewidth=.7)
        axes[1, column].set(xticks=[0, 1, 2], xticklabels=['False safe', 'False reject', 'Joint cost'],
            ylabel='Fitting cost / masked control')
        for row in (0, 1):
            axes[row, column].spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(.5, .94), frameon=False)
    fig.suptitle('Matched causal relations: fitting diagnostics, not safety evidence', y=.98, fontsize=11)
    out = PUBLIC/'figures'; out.mkdir(exist_ok=True)
    path = out/'relation_fitting.svg'; fig.savefig(path, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'relation_fitting.png', dpi=150); plt.close(fig)


def main():
    analysis = source.read(PUBLIC/'analysis.json'); verify = source.read(PUBLIC/'verification.json')
    api.base.exact(verify['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    freeze = source.read(PUBLIC/'training_freeze.json')
    if (freeze['complete_heads'] != 6 or not freeze['all3_masked_controls_exact']
            or not source.read(PUBLIC/'full_refit.json')['exact']
            or not verify['all6_fitting_metrics_and_attributions_exact']):
        raise ValueError('Complete training, matched controls and exact replay required')
    rows = [source.read(source.checked(r)) for r in freeze['fits']]
    matched = comparisons(rows, source.read(run.REG)['jobs'])
    screen = run.fitting_screen([r for r in rows if r['arm'] == 'relational'])
    api.base.exact(screen, analysis['fitting_screen'])
    held = analysis['status'] == 'fresh_run_exposed_development'
    if held != screen['followup_readout_allowed']:
        raise ValueError('Original fitting screen must control readout')
    phases = {k: source.read(PUBLIC/(k+'_execution.json')) for k in ('pilot', 'train', 'evaluate', 'verify')}
    summary = dict(status=analysis['status'], matched=matched, fitting_screen=screen, phases=phases,
        all6_real_fits=True, parameters_per_model=68356, primary_updates=12000, full_refit_updates=2000,
        pilot_updates=500, masked_controls_exact=True, full_refit_exact=True, metric_replay_exact=True,
        parent_assets='cached_verified', models_and_analysis='fresh_run',
        held_readout='fresh_run_exposed_development' if held else 'not_run',
        held_reason=None if held else analysis['reason'], held_policy=analysis.get('held_policy'),
        held_contrasts=analysis.get('contrasts'), original_rules_unchanged=True,
        original24group_primary='not_estimable_not_passed', identity_certification=False,
        independent_confirmation=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'result_summary.json', source.stable(summary)); figure(rows)
    f = lambda x: 'NA' if x is None else f'{x:.4f}'
    text = ['# Causal Rollout-History Feature Experiment', '',
        'I tested whether explicitly relating a candidate forecast to observed motion',
        'helps a neural intervention-risk model make safer decisions. Six models use',
        'identical initialization, query draws, original joint loss and fixed budget.',
        'Both arms have 68,356 parameters; masking the new inputs reduces effective',
        'capacity, so this is not a pure capacity-matched information experiment.', '',
        '## Own-Fitting Results', '',
        '| Context | Arm | Full risk MSE | Joint cost | False-safe | False-reject |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        a = r['joint_attribution']
        text.append(f"| {r['job']} | {r['arm']} | {r['final_fitting']['mean_MSE']:.8f} | "
            f"{a['total_joint_cost']:.8f} | {a['false_safe_count']} | {a['false_reject_count']} |")
    text += ['', '| Context | MSE reduction % | Joint-cost reduction % | False-safe-cost reduction % | False-reject-cost reduction % |',
        '|---|---:|---:|---:|---:|']
    for r in matched:
        text.append('| '+r['job']+' | '+' | '.join(f(r[k+'_reduction_percent']) for k in
            ('mean_MSE', 'total_joint_cost', 'false_safe_cost', 'false_reject_cost'))+' |')
    text += ['', 'Negative reductions mean worse. These are fitting results, not held',
        'trajectory accuracy or a calibrated safety guarantee.', '',
        '![Common loss and decision costs](figures/relation_fitting.svg)', '',
        'Curves use the same sampled batches; full-data metrics above are separate.', '',
        '## Exposed Development', '']
    passed = sum(r['passed'] for r in screen['contexts'])
    text += [f'Original internal fitting screen: **{passed}/3**. Not a deployment gate.', '']
    if held:
        text += ['| Policy | Known ADE gain % | Hard gain % | Switch % | Worst defined easy-net % | Original risk/support |',
            '|---|---:|---:|---:|---:|---:|']
        for name, r in analysis['held_policy'].items():
            s = r['safety']
            text.append('| '+name+' | '+' | '.join(f(x) for x in (r['known']['mean'], r['hard']['mean'],
                r['switch_percent']['mean'], s['worst_defined_signed_easy_net_percent']))+f" | {s['supported']}/{s['required']} |")
        c = analysis['contrasts']['main']
        text += ['', f"Primary relational-minus-masked contrast: {f(c['known_pp']['mean'])} pp; "
            f"missing-label completion bounds [{f(c['completion_lower_pp']['mean'])}, "
            f"{f(c['completion_upper_pp']['mean'])}] pp."]
    else:
        text += ['**not_run:** '+analysis['reason']+'. No held improvement inferred.']
    text += ['', 'Completion bounds are not sampling CIs. Partial-outcome development ADE',
        'is not complete-roster ADE or independent confirmation. One producer seed',
        'and repeatedly exposed nested C cannot support a final paper claim.', '',
        '## Execution', '', '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for key, r in phases.items():
        text.append(f"| {key} | {r['pid']} | {r['seconds']:.2f} | {r['peak_RSS_bytes']} |")
    text += ['', '12,000 primary + 2,000 full-refit + 500 pilot updates. Native arm64 Torch,',
        'CPU4/inter-op1/workers0. Controls, full refit, fitting metrics and error',
        'attributions replay exactly. No new trajectory predictor, JEPA or CREATE job.', '',
        'The original 2% all/easy positive-harm, easy-net, reference/support contract',
        'is unchanged. Empty support remains NA. No identity exemption. The original',
        '24-group primary stays NA; reserved Moving20, DUT, DroneCrowd and H stay closed.',
        'Native obs8/pred12 image coordinates; no metric/seconds/true3D/foundation',
        'or human-gold claim. Stage5C/SMC are disabled; deployment is unchanged.', '',
        '[Protocol](protocol.md) | [Metrics](result_summary.json) |',
        '[Interpretation](failure_analysis.md) | [Reproduction](operation_zh.md)', '']
    (PUBLIC/'README.md').write_text('\n'.join(text))
    risk = analysis['held_policy']['history_relational_veto']['safety']['full_original_risk_pass'] if held else 'not_run'
    (PUBLIC/'gates.md').write_text('\n'.join(['# Evidence Gates', '', '| Gate | Status |', '|---|---|',
        '| Six real fits / original masked controls / resume / full refit | pass |',
        '| All fitting metrics and error attributions replay | pass |',
        f'| Original internal fitting screen | {passed}/3 |', f'| Original held risk/support | {risk} |',
        '| Original 24-group primary | NA, not passed |', '| Independent confirmation | not_run |',
        '| Deployment change / submission readiness | false / false |',
        '| Stage5C / SMC | disabled / disabled |', '',
        'Engineering passes are not pooled with scientific gates.', '']))
    print(source.json.dumps(dict(fits=6, fitting_screen=passed, held_readout=summary['held_readout'],
        original_risk_pass=risk, deployment_changed=False)))


if __name__ == '__main__':
    main()
