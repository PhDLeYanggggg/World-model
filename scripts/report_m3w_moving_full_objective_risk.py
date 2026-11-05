"""Report completed fitting, including a prospectively skipped held readout."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_full_objective_risk as run

source, api, PUBLIC = run.source, run.api, run.PUBLIC


def comparison(row):
    a, b = row['initial_fitting'], row['final_fitting']
    reduction = lambda x, y: 100*(x-y)/x if x > 0 else None
    return dict(job=row['job'], MSE_reduction_percent=reduction(a['mean_MSE'], b['mean_MSE']),
        easy_MSE_reduction_percent=reduction(a['task_MSE'][1], b['task_MSE'][1]),
        sign_regret_reduction_percent=[reduction(x, y) for x, y in zip(a['observed_sign_regret'], b['observed_sign_regret'])],
        false_safe_before=a['false_safe_channels'], false_safe_after=b['false_safe_channels'],
        final_gradient_inf=b['gradient_inf'], stop_reason=row['stop_reason'])


def main():
    result = source.read(PUBLIC/'analysis.json'); replay = source.read(PUBLIC/'verification.json')
    api.base.exact(replay['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    freeze = source.read(PUBLIC/'training_freeze.json')
    if (freeze['complete_heads'] != 3 or not source.read(PUBLIC/'full_refit.json')['exact']
            or not replay['all3_initial_final_fitting_metrics_exact']):
        raise ValueError('Actual full fitting, exact refit and replay required')
    fits = [source.read(source.checked(r)) for r in freeze['fits']]
    for r in fits:
        s = api.base.read_checkpoint(source.checked(r['checkpoint']))
        if s['parameters'] != 32130 or s['changed_parameter_steps'] == 0:
            raise ValueError('Actual neural training required')
        for k in ('initial_fitting', 'final_fitting', 'step', 'changed_parameter_steps', 'full_gradient_passes'):
            api.base.exact(s[k], r[k])
    screen = api.fitting_screen(fits)
    api.base.exact(screen, result['fitting_screen'])
    held = result['status'] == 'fresh_run'
    if held != screen['followup_readout_allowed']:
        raise ValueError('Prospective fitting screen must control readout')
    if held and not replay['all3_predictions_actions_metrics_exact']:
        raise ValueError('Actual held prediction replay required')
    phases = {p: source.read(PUBLIC/(p+'_execution.json')) for p in ('pilot', 'train', 'evaluate', 'verify')}
    summary = dict(status=result['status'], fitting=[comparison(r) for r in fits],
        fitting_screen=screen, held_readout='fresh_run' if held else 'not_run',
        held_readout_reason=None if held else result['reason'], phases=phases,
        primary_actual_changed_steps=freeze['primary_changed_steps'],
        primary_full_gradient_passes=freeze['primary_full_gradient_passes'],
        original_rules_unchanged=True, identity_certification=False,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False,
        full_historical_test_suite_rerun=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'result_summary.json', source.stable(summary))
    lines = ['# Complete-Fitting Neural Optimization', '',
        'I retain the original model, causal features, labels, source roles and',
        '2% risk/support rules, and test complete-fitting optimization. Three',
        'real neural fits and an exact full refit are complete. This is a risk',
        'head diagnostic, not a new trajectory model or a safety certificate.', '',
        '## Fitting Evidence', '',
        '| Context | Mean MSE reduction % | Easy MSE reduction % | All/easy sign-cost reduction % | Final gradient max | Stop |',
        '|---|---:|---:|---|---:|---|']
    for r in summary['fitting']:
        signs = '/'.join('NA' if x is None else f'{x:.3f}' for x in r['sign_regret_reduction_percent'])
        lines.append(f"| {r['job']} | {r['MSE_reduction_percent']:.3f} | {r['easy_MSE_reduction_percent']:.3f} | {signs} | {r['final_gradient_inf']:.5g} | {r['stop_reason']} |")
    lines += ['', 'Positive reductions mean improvement on fitting data, not held safety.',
        'These losses predict normalized signed all/easy budget-excess risk;',
        'they are not trajectory ADE/FDE. Unknown labels retain original',
        'conservative bounds. Sign costs use observed fitting labels only.', '',
        f"Prospective fitting expansion screen: **{screen['followup_readout_allowed']}**.",
        'The fixed screen requires >=1% mean-MSE reduction and non-increasing',
        'cost-weighted sign errors in BOTH channels in EVERY fitting context.',
        'It does not change or replace any original evaluation rule.', '',
        '## Held Evidence', '']
    if not held:
        lines += ['**not_run:** '+result['reason']+'.',
            'No new held utility or safety result is inferred from fitting losses.',
            'Earlier held failures remain failures. There is no deployment upgrade.']
    else:
        lines += ['All fits froze before the same already exposed nested-C readout.',
            '| Policy | Known floor gain % | Hard gain % | Switch % | Full risk/support | Worst easy-net % |',
            '|---|---:|---:|---:|---:|---:|']
        fmt = lambda x: 'NA' if x is None else f'{x:.4f}'
        for name in ('readout_veto', 'constant_veto', 'precision_only_veto', 'full_objective_veto'):
            p = result['held_policy'][name]; s = p['safety']
            lines.append(f"| {name} | {fmt(p['known']['mean'])} | {fmt(p['hard']['mean'])} | {fmt(p['switch_percent']['mean'])} | {s['supported']}/{s['required']} | {fmt(s['worst_defined_signed_easy_net_percent'])} |")
        lines += ['', '| Contrast | Known pp | Completion lower pp | Completion upper pp |', '|---|---:|---:|---:|']
        for name, r in result['contrasts'].items():
            lines.append('| '+name+' | '+' | '.join(fmt(r[k]['mean']) for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp'))+' |')
        lines += ['', 'Completion bounds are NOT sampling CIs. Partial known ADE is not',
            'complete-roster 12-step ADE. These repeated development observations',
            'are not independent confirmation. The float64 initial control separates',
            'precision from subsequent fitting; old cached actions are hash-verified',
            'and their metrics recomputed exactly. No old control was freshly trained.']
    lines += ['', '## Computation And Replay', '',
        '| Context | Outer calls | Parameter-changing calls | Full gradient passes | LBFGS function evaluations | Seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for r in fits:
        lines.append(f"| {r['job']} | {r['step']} | {r['changed_parameter_steps']} | {r['full_gradient_passes']} | {r['lbfgs_function_evaluations']} | {r['seconds']:.2f} |")
    lines += ['', '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for name, r in phases.items():
        lines.append(f"| {name} | {r['pid']} | {r['seconds']:.2f} | {r['peak_RSS_bytes']} |")
    lines += ['', 'The evaluate phase may only record not_run; its successful exit does',
        'not imply held evaluation took place. Full-gradient passes include initial',
        'and final-gradient diagnostics and repeated line-search evaluations.',
        'Reaching200 outer calls is not proof of convergence.', '',
        'Each context passes actual3+3 versus6 resume equivalence. The first full',
        'fit repeats exactly; all initial/final full-fitting metrics replay exactly.',
        'Interrupted line searches cannot overwrite the last completed checkpoint.',
        'This solver/float64/no-decay bundle is not an isolated optimizer comparison.', '',
        'Original24-group primary: NA, not passed. Missing support and empty',
        'controls remain NA, without identity certification. No new H/reserved data.',
        'Native obs8/pred12 image pixels; no metric/seconds/true3D/foundation or',
        'human-gold claim. Stage5C and SMC remain disabled.', '',
        '[Protocol](protocol.md) | [Fitting screen](fitting_screen.json) | [Analysis](analysis.json)', '']
    (PUBLIC/'README.md').write_text('\n'.join(lines))
    commands = '\n'.join('.venv-pytorch/bin/python scripts/run_m3w_moving_full_objective_risk.py '+p
        for p in ('register', 'pilot', 'train', 'evaluate', 'verify'))
    (PUBLIC/'reproducibility.md').write_text('# Reproducibility\n\n```sh\n'+commands+'\n```\n\n'
        'Use unchanged --resume only after an interrupted phase. Completed phases\n'
        'refuse overwrite. Checkpoints and row caches remain private and must match\n'
        'recorded hashes. The first complete fit is independently rerun exactly;\n'
        'this is not an independent data test. Regenerate reports with\n'
        'scripts/report_m3w_moving_full_objective_risk.py.\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
