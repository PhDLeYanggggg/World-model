"""Evidence report for a frozen schedule experiment, never a result selector."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_annealed_risk as run

source, api, PUBLIC = run.source, run.api, run.PUBLIC


def summary(result):
    p = result['held_policy']; a, b = p['constant_veto'], p['cosine_veto']
    c = result['contrasts']['cosine_minus_constant']
    safety = b['safety']
    return dict(main_contrast={k: v['mean'] for k, v in c.items()},
        known_gain_before=a['known']['mean'], known_gain_after=b['known']['mean'],
        hard_gain_before=a['hard']['mean'], hard_gain_after=b['hard']['mean'],
        easy_worst_before=a['safety']['worst_defined_signed_easy_net_percent'],
        easy_worst_after=safety['worst_defined_signed_easy_net_percent'],
        full_recording_risk_support_pass=safety['supported'] == safety['required'] and safety['required'] > 0,
        expansion_supported=result['expansion_supported'],
        original24group_primary=result['original24group_primary'],
        completion_bounds_are_sampling_CI=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def main():
    result = source.read(PUBLIC/'analysis.json'); replay = source.read(PUBLIC/'verification.json')
    api.base.exact(replay['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    if not replay['all3_predictions_actions_metrics_exact'] or not replay['old_cached_control_metrics_exact']:
        raise ValueError('Actual inference, action and metric replay required')
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_heads'] != 6 or not source.read(PUBLIC/'full_refit.json')['exact']:
        raise ValueError('All fits and actual full treatment refit required')
    fits = [source.read(source.checked(ref)) for ref in freeze['fits']]
    for r in fits:
        state = api.base.read_checkpoint(source.checked(r['checkpoint']))
        if state['step'] != 2000 or state['parameters'] != 32130:
            raise ValueError('Full actual neural budget required')
        if any(int(v['step']) != 2000 for v in state['optimizer']['state'].values()):
            raise ValueError('Full optimizer updates required')
        if api.torch.equal(state['model']['encoder.weight'], state['initial_model']['encoder.weight']):
            raise ValueError('Real encoder updates required')
    s = summary(result)
    phases = {p: source.read(PUBLIC/(p+'_execution.json')) for p in ('pilot', 'train', 'evaluate', 'verify')}
    source.once(PUBLIC/'result_summary.json', source.stable(dict(s, phases=phases,
        source='fresh_report_from_verified_run_artifacts', analysis=source.artifact(PUBLIC/'analysis.json'),
        primary_updates=12000, full_refit_updates=2000, resume_check_updates_including_repeats=400,
        full_historical_test_suite_rerun=False)))
    fmt = lambda v: 'NA' if v is None else f'{v:.4f}'
    lines = ['# Fixed-Schedule Neural Risk Training', '',
        'I compare constant and gradually reduced learning rates while retaining',
        'the original model, causal inputs, labels, loss, update budget and safety',
        'rules. Six real neural fits and a full treatment refit are complete.',
        'No thresholds or evaluation support rules were changed.', '',
        '## Held Development Evidence', '',
        '| Model | Known ADE gain over floor % | Hard gain % | Intervention % | Full risk/support | Worst defined easy-net % |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('readout_veto', 'kinematic_veto', 'constant_veto', 'cosine_veto'):
        p = result['held_policy'][name]; risk = p['safety']
        lines.append(f"| {name} | {fmt(p['known']['mean'])} | {fmt(p['hard']['mean'])} | {fmt(p['switch_percent']['mean'])} | {risk['supported']}/{risk['required']} | {fmt(risk['worst_defined_signed_easy_net_percent'])} |")
    lines += ['', '| Contrast | Known pp | Completion lower pp | Completion upper pp |',
        '|---|---:|---:|---:|']
    for name, c in result['contrasts'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(c[k]['mean']) for k in
            ('known_pp', 'completion_lower_pp', 'completion_upper_pp'))+' |')
    lines += ['', '**Completion bounds are not sampling confidence intervals.**',
        'These are repeated development results on seven SOURCE units and one',
        'producer seed, not independent confirmation. Partial observed ADE is not',
        'complete-roster 12-step ADE. The original 24-group primary remains NA.', '',
        '## Original Safety Checks', '',
        '| Schedule | Known all-risk violations | Known easy-risk violations | Undefined selected easy | Undefined full easy |',
        '|---|---:|---:|---:|---:|']
    for name in api.ARMS:
        r = result['held_policy'][name+'_veto']['safety']
        lines.append('| '+name+' | '+' | '.join(str(r[k]) for k in
            ('known_all_violations', 'known_easy_violations', 'undefined_selected_easy', 'undefined_full_easy_net'))+' |')
    lines += ['', f"Development expansion screen: {s['expansion_supported']}.",
        f"Full recording risk/support gate: {s['full_recording_risk_support_pass']}.",
        'A local improvement cannot replace the full safety gate. Empty controls',
        'and missing support remain NA, without an identity-fallback certificate.', '',
        '## Whole-Fitting Diagnostics', '',
        'Computed before held prediction, over all own-B queries with the same',
        'empirical sampling weights. Lower training loss is not held-scene safety.', '',
        'These losses measure normalized signed budget-excess risk prediction,',
        'not trajectory ADE/FDE. Unknown outcomes retain conservative bound',
        'supervision, rather than being silently dropped.', '',
        '| Context | Schedule | All MSE | Easy MSE | All/easy sign regret | False-safe all/easy |',
        '|---|---|---:|---:|---|---|']
    for f in fits:
        a = f['full_fitting']
        lines.append(f"| {f['job']} | {f['arm']} | {a['task_MSE'][0]:.8g} | {a['task_MSE'][1]:.8g} | {a['observed_sign_regret']} | {a['false_safe_channels']} |")
    lines += ['', '## Runtime And Reproducibility', '', '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for name, p in phases.items():
        lines.append(f"| {name} | {p['pid']} | {p['seconds']:.2f} | {p['peak_RSS_bytes']} |")
    lines += ['', 'Native arm64 Torch CPU4/inter-op1/workers0. Both schedules resume',
        'exactly; the first full cosine refit matches exactly. Three fresh constant',
        'controls reproduce old weights, optimizer and sample draws. All new held',
        'predictions, actions and metrics replay exactly; old cached actions are',
        'hash-verified and their metrics recomputed. All six fits froze before V.',
        'These are engineering and development checks, not independent test results.', '',
        'No new H/reserved data, deployment upgrade, trajectory-dynamics training',
        'or JEPA pretraining. Native obs8/pred12 image pixels: no metric, seconds,',
        'true3D, foundation or human-gold claim. Stage5C and SMC remain disabled.', '',
        '[Protocol](protocol.md) | [Analysis](analysis.json) | [Replay](verification.json)', '']
    (PUBLIC/'README.md').write_text('\n'.join(lines))
    lines = ['# All Development Sources', '',
        '| Source | Known contrast pp | Completion lower pp | Completion upper pp | Constant switch % | Cosine switch % |',
        '|---|---:|---:|---:|---:|---:|']
    c = result['contrasts']['cosine_minus_constant']
    for site in sorted(c['known_pp']['by_source']):
        values = [c[k]['by_source'][site] for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')]
        values += [result['held_policy'][a+'_veto']['switch_percent']['by_source'][site] for a in api.ARMS]
        lines.append('| '+site+' | '+' | '.join(fmt(v) for v in values)+' |')
    lines += ['', 'Completion bounds are not sampling CIs. No source is dropped or',
        'selected post hoc. Whole-policy original support checks remain required.', '']
    (PUBLIC/'per_source.md').write_text('\n'.join(lines))
    commands = '\n'.join('.venv-pytorch/bin/python scripts/run_m3w_moving_annealed_risk.py '+p
        for p in ('register', 'pilot', 'train', 'evaluate', 'verify'))
    (PUBLIC/'reproducibility.md').write_text('# Reproducibility\n\n```sh\n'+commands+'\n```\n\n'
        'On original completed outputs, phases refuse overwrite. Use unchanged --resume\n'
        'only for interrupted phases. The pilot verifies exact resumed weights.\n'
        'Reports can be regenerated with report_m3w_moving_annealed_risk.py.\n'
        'Checkpoints and caches are private and must match the recorded hashes.\n')
    print(json.dumps(s, indent=2))


if __name__ == '__main__':
    main()
