"""Report every fixed harm-allocation arm without selecting from held results."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_harm_floor as run
from scripts.report_m3w_moving_decoupled_risk import compare, reduction

source, api, PUBLIC = run.source, run.api, run.PUBLIC


def verify_roster(rows, jobs):
    expected = {(j['key'], arm) for j in jobs for arm in api.ARMS}
    actual = [(r['job'], r['arm']) for r in rows]
    if (len(jobs) != 3 or len(rows) != 9 or len(set(actual)) != 9 or set(actual) != expected
            or any(r['step'] != 2000 or r['parameters'] != 64260 or not r['both_neural_encoders_updated'] for r in rows)):
        raise ValueError('All nine unique completed fits required')


def figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.hashsalt': 'moving_harm_floor_v1', 'svg.fonttype': 'none', 'font.size': 9})
    import matplotlib.pyplot as plt
    jobs = sorted({r['job'] for r in rows}, key=lambda x: int(x.rsplit('critic', 1)[1]))
    colors = {'joint': '#555555', 'uniform_harm': '#2878A0', 'harm_floor': '#B84848'}
    public = PUBLIC/'figures'; public.mkdir(exist_ok=True)
    for mode in ('common_risk_loss', 'decision_costs'):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
        for ax, job in zip(axes, jobs):
            arms = {r['arm']: r for r in rows if r['job'] == job}
            if mode == 'common_risk_loss':
                for arm in api.ARMS:
                    trace = arms[arm]['trace']
                    ax.plot([t['step'] for t in trace], [t['risk_MSE'] for t in trace],
                        color=colors[arm], label=arm, linewidth=1.2)
                ax.set(xlabel='Optimizer update', ylabel='Common risk MSE (sampled batches)')
            else:
                metrics = ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')
                for i, arm in enumerate(api.ARMS):
                    values = [arms[arm]['joint_attribution']['total'][k] /
                        arms['joint']['joint_attribution']['total'][k] for k in metrics]
                    ax.bar(source.np.arange(3)+(i-1)*.24, values, width=.24, color=colors[arm], label=arm)
                ax.axhline(1, color='#333333', linewidth=.7, linestyle='--')
                ax.set(xticks=[0, 1, 2], xticklabels=['False safe', 'False reject', 'Joint cost'],
                    ylabel='Own-fitting cost / joint control (lower is better)')
            ax.set_title(job.rsplit('_', 1)[1]); ax.spines[['top', 'right']].set_visible(False)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle('Fitting diagnostics only: not held utility or safety certification', fontsize=11)
        svg = public/(mode+'.svg'); fig.savefig(svg, metadata={'Date': None})
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        fig.savefig(run.PRIVATE/(mode+'.png'), dpi=150); plt.close(fig)


def main():
    a = source.read(PUBLIC/'analysis.json'); v = source.read(PUBLIC/'verification.json')
    api.base.exact(v['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    freeze = source.read(PUBLIC/'training_freeze.json')
    if (freeze['complete_heads'] != 9 or not freeze['all3_controls_reproduce_previous']
            or not v['all9_initial_final_fitting_metrics_exact'] or not v['all9_joint_attributions_exact']
            or not source.read(PUBLIC/'full_refit.json')['exact']
            or not source.read(PUBLIC/'weighting_verification.json')['all9_exact']):
        raise ValueError('Actual complete fits, controls, refit, weights and replay required')
    rows = [source.read(source.checked(ref)) for ref in freeze['fits']]
    verify_roster(rows, source.read(run.REG)['jobs'])
    figures(rows)
    screen = run.fitting_screen([r for r in rows if r['arm'] == 'harm_floor'])
    api.base.exact(screen, a['fitting_screen'])
    held = a['status'] == 'fresh_run'
    if held != screen['followup_readout_allowed']:
        raise ValueError('Prespecified screen controls readout')
    matched = []
    for job in sorted({r['job'] for r in rows}):
        arms = {r['arm']: r for r in rows if r['job'] == job}
        for left, right in (('joint', 'uniform_harm'), ('joint', 'harm_floor'), ('uniform_harm', 'harm_floor')):
            x, y = arms[left]['joint_attribution']['total'], arms[right]['joint_attribution']['total']
            matched.append(dict(job=job, comparison=right+'_minus_'+left,
                joint_cost_reduction_percent=reduction(x['total_joint_cost'], y['total_joint_cost']),
                false_safe_cost_reduction_percent=reduction(x['false_safe_cost'], y['false_safe_cost']),
                false_reject_cost_reduction_percent=reduction(x['false_reject_cost'], y['false_reject_cost'])))
    phases = {key: source.read(PUBLIC/(key+'_execution.json')) for key in ('pilot', 'train', 'evaluate', 'verify')}
    summary = dict(status=a['status'], fitting=[compare(r) for r in rows], matched_contrasts=matched,
        fitting_harm_weighting={r['job']: r['fitting_harm_weighting'] for r in rows},
        fitting_screen=screen, phases=phases, primary_updates=18000, full_refit_updates=2000, pilot_updates=600,
        training='fresh_run_complete', controls_exact=True, full_refit_exact=True, all9_weights_exact=True,
        all9_metrics_and_attributions_exact=True,
        held_readout='fresh_run_exposed_development' if held else 'not_run',
        held_readout_reason=None if held else a['reason'],
        held_policy=a['held_policy'] if held else None, held_contrasts=a['contrasts'] if held else None,
        completion_bounds_are_sampling_CI=False, original_rules_unchanged=True, identity_certification=False,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    first = next(r for r in rows if r['job'] == 'eqmotion_fold1_seed17_critic0' and r['arm'] == 'joint')
    if first['checkpoint']['sha256'] != 'e4bad40a0b2f320f87d97c43fc00697765d56fb99e638d8f48a64dae471858fd':
        raise ValueError('Recovered first checkpoint must remain byte-identical')
    summary['recovery'] = dict(first_attempt_pid=9913, first_attempt_exit_code=1,
        phase='receipt_serialization_after_first_complete_control',
        original_checkpoint_sha256=first['checkpoint']['sha256'], checkpoint_bytes_unchanged=True,
        first_fit_seconds=first['seconds'], completed_train_receipt_excludes_failed_attempt_wall_time=True,
        serialization_amendment=source.artifact(run.AMENDMENT))
    source.once(PUBLIC/'result_summary.json', source.stable(summary))
    fmt = lambda x: 'NA' if x is None else f'{x:.4f}'
    text = ['# Harm-Weight Allocation Under Unchanged Safety Rules', '',
        'I tested whether strengthening lower-magnitude harmful labels improves',
        'the joint intervention decision. A uniform-weight arm has the same total',
        'positive coefficient mass, separating allocation from a uniform increase.',
        'All nine full fits, exact controls, recovery checks and the first full',
        'treatment refit are complete. These are risk heads, not new forecasts.', '',
        '## Fitting Results', '',
        '| Context | Arm | MSE reduction from warm % | All sign-cost reduction % | Easy sign-cost reduction % | False-safe proposals | False rejections |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        m, c = r['joint_attribution']['total'], compare(r)
        text.append(f"| {r['job']} | {r['arm']} | {fmt(c['MSE_reduction_percent'])} | "
            f"{fmt(c['sign_regret_reduction_percent'][0])} | {fmt(c['sign_regret_reduction_percent'][1])} | "
            f"{m['false_safe_count']} | {m['false_reject_count']} |")
    text += ['', '| Context | Contrast | Joint-cost reduction % | False-safe-cost reduction % | False-reject-cost reduction % |',
        '|---|---|---:|---:|---:|']
    for r in matched:
        text.append(f"| {r['job']} | {r['comparison']} | "+' | '.join(fmt(r[k]) for k in
            ('joint_cost_reduction_percent', 'false_safe_cost_reduction_percent', 'false_reject_cost_reduction_percent'))+' |')
    text += ['', 'Negative reductions mean worse. Counts and normalized costs are fitting',
        'diagnostics, not held trajectory gains or deployment guarantees.', '',
        '![Common risk loss, same query draws](figures/common_risk_loss.svg)', '',
        'Different auxiliary weights make total training objectives incomparable',
        'across arms. The plotted risk MSE is unchanged; sampled-batch values are',
        'not full-population fitting loss. Curves may overlap.', '',
        '![False-safe and false-rejection fitting costs](figures/decision_costs.svg)', '',
        '| Context | Positive weighted median | Uniform multiplier | Original positive mass | New positive mass |',
        '|---|---:|---:|---:|---:|']
    for job, w in summary['fitting_harm_weighting'].items():
        text.append('| '+job+' | '+' | '.join(fmt(w[k]) for k in
            ('positive_weighted_median', 'uniform_multiplier', 'original_positive_mass', 'floor_positive_mass'))+' |')
    text += ['', 'Loss-weight magnitudes use the mean positive target across channels.',
        'Attributed decision costs above sum channel magnitudes. Their absolute',
        'values are not interchangeable; matched comparisons retain one convention.']
    passed = sum(r['passed'] for r in screen['contexts'])
    text += ['', f'Original internal fitting screen: **{passed}/3**, overall **{screen["followup_readout_allowed"]}**.',
        'It compares harm_floor with warm initialization, not a scientific risk',
        'gate or a posthoc choice among fitted arms.', '', '## Exposed Development', '']
    if held:
        text += ['Seven previously exposed SOURCE units / 17 recording views, one producer seed.', '',
            '| Policy | Known ADE gain % | Hard gain % | Switch % | Worst defined easy-net % | Full risk/support |',
            '|---|---:|---:|---:|---:|---:|']
        for name in ('constant_veto', 'joint_veto', 'uniform_harm_veto', 'harm_floor_veto', 'harm_floor_risk_only_veto'):
            r = a['held_policy'][name]; s = r['safety']
            text.append('| '+name+' | '+' | '.join(fmt(x) for x in
                (r['known']['mean'], r['hard']['mean'], r['switch_percent']['mean'], s['worst_defined_signed_easy_net_percent']))+
                f" | {s['supported']}/{s['required']} |")
        r = a['contrasts']['main']; s = a['held_policy']['harm_floor_veto']['safety']
        text += ['', f"Primary harm_floor minus uniform_harm: {fmt(r['known_pp']['mean'])} pp, "
            f"completion bounds [{fmt(r['completion_lower_pp']['mean'])}, {fmt(r['completion_upper_pp']['mean'])}] pp.",
            f"Original full risk/support pass: **{s['full_original_risk_pass']}**. "
            f"Known all/easy violations: {s['known_all_violations']}/{s['known_easy_violations']}; "
            f"undefined selected-easy views: {s['undefined_selected_easy']}.",
            'Completion bounds are not sampling CIs. Partial-outcome known ADE is',
            'not complete-roster 12-step ADE; exposed development is not independent confirmation.']
    else:
        text += ['**not_run:** '+a['reason']+'. No new held predictions or scientific gain inferred.']
    text += ['', '## Runtime', '', '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for key, p in phases.items():
        text.append(f"| {key} | {p['pid']} | {p['seconds']:.2f} | {p['peak_RSS_bytes']} |")
    text += ['', 'Native arm64 PyTorch CPU 4/inter-op 1/workers 0. Checkpoints/heartbeat',
        'every 100 completed updates; 18,000 primary, 2,000 refit and 600 pilot updates.',
        'Own-B medians, mass controls and all sampler counts recompute exactly.', '',
        'The training phase above is the resumed process. The first attempt exited1',
        'at JSON receipt export after completing the first control. Its checkpoint',
        f"is byte-identical after recovery; its {first['seconds']:.2f} fit seconds are",
        'not included in resumed process time. No completed optimizer steps were',
        'repeated. The original registration and failed attempt remain recorded.', '',
        'Parent assets cached_verified; training/diagnostics fresh_run. Reserved/H',
        'confirmation not_run. Original 2% rules unchanged, no identity exemption.',
        'Empty controls/support stay NA; original 24-group primary remains NA, not pass.',
        'No deployment or submission-ready claim. Native obs8/pred12 image coordinates;',
        'not metric, seconds, true 3D, foundation or human gold. Stage5C and SMC disabled.', '',
        '[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) |',
        '[Run/recovery guide](operation_zh.md) | [Numbers](result_summary.json)', '']
    (PUBLIC/'README.md').write_text('\n'.join(text))
    safety = a['held_policy']['harm_floor_veto']['safety']['full_original_risk_pass'] if held else 'not_run'
    (PUBLIC/'gates.md').write_text('\n'.join(['# Evidence Gates', '', '| Gate | Status |', '|---|---|',
        '| Nine real fits / exact predecessor controls | pass |', '| Resume / full refit / own-B weights replay | pass |',
        '| Complete fitting metrics and attribution replay | pass |',
        f'| Original internal fitting screen | {passed}/3 |', f'| Original held risk/support | {safety} |',
        '| Original 24-group primary | NA, not passed |', '| Independent confirmation | not_run |',
        '| Deployment / submission readiness | false / false |', '| Stage5C / SMC | disabled / disabled |', '',
        'Execution checks are not scientific successes and are not pooled into a gate total.', '']))
    print(json.dumps(dict(status=summary['status'], completed_fits=len(rows),
        fitting_screen_passed=passed, held_readout=summary['held_readout'], deployment_changed=False)))


if __name__ == '__main__':
    main()
