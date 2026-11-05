"""Complete, verified paired-loss results; no selection from held outcomes."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_paired_decision as run
from scripts.report_m3w_moving_decoupled_risk import compare, reduction

source, api, PUBLIC = run.source, run.api, run.PUBLIC


def verify_roster(rows, jobs):
    expected = {(j['key'], arm) for j in jobs for arm in api.ARMS}
    actual = [(r['job'], r['arm']) for r in rows]
    if (len(jobs) != 3 or len(rows) != 9 or len(set(actual)) != 9 or set(actual) != expected
            or any(r['step'] != 2000 or r['parameters'] != 64260 or not r['both_neural_encoders_updated'] for r in rows)):
        raise ValueError('All nine unique complete fits required')


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.hashsalt': 'moving_paired_decision_v1', 'svg.fonttype': 'none', 'font.size': 9})
    import matplotlib.pyplot as plt
    colors = {'joint': '#555555', 'paired_point': '#2878A0', 'paired_rank': '#B84848'}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.subplots_adjust(left=.075, right=.985, bottom=.08, top=.84, hspace=.5, wspace=.4)
    jobs = sorted({r['job'] for r in rows}, key=lambda x: int(x.rsplit('critic', 1)[1]))
    for column, job in enumerate(jobs):
        arms = {r['arm']: r for r in rows if r['job'] == job}
        for arm in api.ARMS:
            trace = arms[arm]['trace']
            axes[0, column].plot([r['step'] for r in trace], [r['risk_MSE'] for r in trace],
                color=colors[arm], label=arm, linewidth=1.1)
        axes[0, column].set(title=job.rsplit('_', 1)[1], xlabel='Optimizer update', ylabel='Common sampled-batch risk MSE')
        keys = ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')
        for i, arm in enumerate(api.ARMS):
            ratios = [arms[arm]['joint_attribution']['total'][k]/arms['joint']['joint_attribution']['total'][k] for k in keys]
            axes[1, column].bar(source.np.arange(3)+(i-1)*.24, ratios, width=.24, color=colors[arm])
        axes[1, column].axhline(1, linestyle='--', color='#333333', linewidth=.7)
        axes[1, column].set(xticks=[0, 1, 2], xticklabels=['False safe', 'False reject', 'Joint cost'], ylabel='Fitting cost / joint control')
        for row in (0, 1):
            axes[row, column].spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(.5, .94), frameon=False)
    fig.suptitle('Fitting diagnostics only: not independent utility or safety evidence', y=.98, fontsize=11)
    out = PUBLIC/'figures'; out.mkdir(exist_ok=True)
    path = out/'paired_fitting.svg'; fig.savefig(path, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'paired_fitting.png', dpi=150); plt.close(fig)


def main():
    analysis = source.read(PUBLIC/'analysis.json'); verification = source.read(PUBLIC/'verification.json')
    api.base.exact(verification['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    freeze = source.read(PUBLIC/'training_freeze.json')
    if (freeze['complete_heads'] != 9 or not freeze['all3_controls_reproduce_previous']
            or not source.read(PUBLIC/'full_refit.json')['exact']
            or not all(verification[k] for k in ('all9_initial_final_fitting_metrics_exact',
                'all9_joint_attributions_exact', 'all9_discrimination_exact'))):
        raise ValueError('Complete training and real exact replay required')
    rows = [source.read(source.checked(r)) for r in freeze['fits']]
    verify_roster(rows, source.read(run.REG)['jobs']); figure(rows)
    screen = run.fitting_screen([r for r in rows if r['arm'] == 'paired_rank'])
    api.base.exact(screen, analysis['fitting_screen'])
    held = analysis['status'] == 'fresh_run'
    if held != screen['followup_readout_allowed']:
        raise ValueError('Original fitting screen controls readout')
    matched = []
    for job in sorted({r['job'] for r in rows}):
        arms = {r['arm']: r for r in rows if r['job'] == job}
        for left, right in (('joint', 'paired_point'), ('joint', 'paired_rank'), ('paired_point', 'paired_rank')):
            a, b = (arms[name]['joint_attribution']['total'] for name in (left, right))
            matched.append(dict(job=job, comparison=right+'_minus_'+left,
                **{k+'_reduction_percent': reduction(a[k], b[k]) for k in
                    ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')}))
    phases = {key: source.read(PUBLIC/(key+'_execution.json')) for key in ('pilot', 'train', 'evaluate', 'verify')}
    summary = dict(status=analysis['status'], fits=[compare(r) for r in rows], matched=matched,
        discrimination=[dict(job=r['job'], arm=r['arm'], **r['discrimination']) for r in rows],
        phases=phases, fitting_screen=screen, primary_updates=18000, full_refit_updates=2000, pilot_updates=600,
        all9_real_fits=True, exact_controls=True, exact_full_refit=True, exact_replay=True,
        held_readout='fresh_run_exposed_development' if held else 'not_run',
        held_reason=None if held else analysis['reason'], held_policy=analysis.get('held_policy'),
        held_contrasts=analysis.get('contrasts'), completion_bounds_are_sampling_CI=False,
        independent_confirmation=False, original_rules_unchanged=True, identity_certification=False,
        original24group_primary='not_estimable_not_passed', deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'result_summary.json', source.stable(summary))
    f = lambda x: 'NA' if x is None else f'{x:.4f}'
    text = ['# Paired Useful/Harmful Decision Learning', '',
        'I tested within-query paired ranking against a matched pointwise loss and',
        'the preceding joint control. All nine complete fits, three exact controls,',
        'one full treatment refit and fitting-metric replay are complete.',
        'Held evaluation and its replay status are reported separately below.',
        'These are neural intervention-risk heads, not new trajectory forecasts.', '',
        '## Own-Fitting Results', '',
        '| Context | Arm | MSE reduction from warm % | Joint cost | False-safe | False-reject | AUROC | Cost-weighted AUROC |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        a, d = r['joint_attribution']['total'], r['discrimination']['total']['combined']
        text.append(f"| {r['job']} | {r['arm']} | {f(compare(r)['MSE_reduction_percent'])} | {a['total_joint_cost']:.8f} | "
            f"{a['false_safe_count']} | {a['false_reject_count']} | {f(d['query_weighted_AUROC'])} | {f(d['cost_weighted_AUROC'])} |")
    text += ['', '| Context | Contrast | Joint-cost reduction % | False-safe-cost reduction % | False-reject-cost reduction % |',
        '|---|---|---:|---:|---:|']
    for r in matched:
        text.append('| '+r['job']+' | '+r['comparison']+' | '+' | '.join(f(r[k+'_reduction_percent'])
            for k in ('total_joint_cost', 'false_safe_cost', 'false_reject_cost'))+' |')
    text += ['', 'Negative reductions mean worse. Ranking excludes zero-target and unknown',
        'rows; attribution counts use the preceding target-safe convention.',
        'Higher fitting AUROC is not better held decisions or a safety certificate.', '',
        '![Common loss and decision costs](figures/paired_fitting.svg)', '',
        'The common MSE term is comparable; differently constructed total objectives',
        'are not interchangeable. Sampled-batch curves may overlap.', '', '## Exposed Development', '']
    passed = sum(r['passed'] for r in screen['contexts'])
    text += [f'Original internal fitting screen: **{passed}/3**. It is not the scientific safety gate.', '']
    if held:
        text += ['Seven already exposed SOURCE units / 17 views, one producer seed.', '',
            '| Policy | Known ADE gain % | Hard gain % | Switch % | Worst defined easy-net % | Full risk/support |',
            '|---|---:|---:|---:|---:|---:|']
        for name in ('constant_veto', 'joint_veto', 'paired_point_veto', 'paired_rank_veto', 'paired_rank_risk_only_veto'):
            r = analysis['held_policy'][name]; s = r['safety']
            text.append('| '+name+' | '+' | '.join(f(x) for x in (r['known']['mean'], r['hard']['mean'],
                r['switch_percent']['mean'], s['worst_defined_signed_easy_net_percent']))+f" | {s['supported']}/{s['required']} |")
        c = analysis['contrasts']['main']; s = analysis['held_policy']['paired_rank_veto']['safety']
        text += ['', f"Primary rank-minus-point known contrast: {f(c['known_pp']['mean'])} pp; completion bounds "
            f"[{f(c['completion_lower_pp']['mean'])}, {f(c['completion_upper_pp']['mean'])}] pp.",
            f"Original full risk/support pass: **{s['full_original_risk_pass']}**; "
            f"known all/easy violations {s['known_all_violations']}/{s['known_easy_violations']}; "
            f"undefined selected-easy views {s['undefined_selected_easy']}."]
    else:
        text += ['**not_run:** '+analysis['reason']+'. No held trajectory improvement inferred.']
    text += ['', 'Completion bounds are not sampling CIs. Known partial-outcome ADE is not',
        'complete-roster ADE. Repeated development is not independent confirmation.', '',
        '## Execution', '', '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for k, r in phases.items():
        text.append(f"| {k} | {r['pid']} | {r['seconds']:.2f} | {r['peak_RSS_bytes']} |")
    text += ['', '18,000 primary, 2,000 full-refit and 600 pilot updates; actual arm64 PyTorch',
        'CPU4/inter-op1/workers0. Parent assets cached_verified, fits/diagnostics',
        'fresh_run, reserved/H confirmation not_run. No CREATE job started.',
        'The train-phase duration is for the resumed process only, excluding',
        'initial-process time and downtime. The [interruption and recovery record](interruption_recovery.md)',
        'retains the unavailable first exit code; no crash cause is inferred.', '',
        'Original 2% risk/easy/support rules unchanged; no identity certification.',
        'Original 24-group primary remains NA; empty support is not passed.',
        'No deployment or submission-ready claim. Native obs8/pred12 image',
        'coordinates, not metric/seconds/true3D/foundation/human gold. Stage5C/SMC disabled.', '',
        '[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) |',
        '[Loss geometry](loss_geometry.md) | [Validation receipt](validation_receipt.md) |',
        '[Run/recovery guide](operation_zh.md) | [Metrics](result_summary.json)', '']
    (PUBLIC/'README.md').write_text('\n'.join(text))
    risk = analysis['held_policy']['paired_rank_veto']['safety']['full_original_risk_pass'] if held else 'not_run'
    (PUBLIC/'gates.md').write_text('\n'.join(['# Evidence Gates', '', '| Gate | Status |', '|---|---|',
        '| Nine real fits / exact controls / resume / full refit | pass |',
        '| All fitting metrics / attribution / ranking replay | pass |',
        f'| Original internal fitting screen | {passed}/3 |', f'| Original held risk/support | {risk} |',
        '| Original 24-group primary | NA, not passed |', '| Independent confirmation | not_run |',
        '| Deployment / submission readiness | false / false |', '| Stage5C / SMC | disabled / disabled |', '',
        'Execution passes are not scientific success and are not pooled into a gate total.', '']))
    print(json.dumps(dict(status=summary['status'], fits=9, fitting_screen=passed,
        held_readout=summary['held_readout'], original_risk_pass=risk, deployment_changed=False)))


if __name__ == '__main__':
    main()
