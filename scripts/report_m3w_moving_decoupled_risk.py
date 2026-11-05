"""Render all fixed contrasts without choosing a winning checkpoint or threshold."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_decoupled_risk as run

source, api, PUBLIC = run.source, run.api, run.PUBLIC


def reduction(before, after):
    return 100*(before-after)/before if before > 0 else None


def compare(r):
    a, b = r['initial_fitting'], r['final_fitting']
    return dict(job=r['job'], arm=r['arm'], parameters=r['parameters'], steps=r['step'], seconds=r['seconds'],
        MSE_reduction_percent=reduction(a['mean_MSE'], b['mean_MSE']),
        easy_MSE_reduction_percent=reduction(a['task_MSE'][1], b['task_MSE'][1]),
        sign_regret_reduction_percent=[reduction(x, y) for x, y in zip(a['observed_sign_regret'], b['observed_sign_regret'])],
        risk_only_sign_regret=b['risk_only_sign_regret'], decision_only_sign_regret=b['decision_only_sign_regret'],
        combined_sign_regret=b['observed_sign_regret'],
        false_safe_before=a['false_safe_channels'], false_safe_after=b['false_safe_channels'])


def main():
    a = source.read(PUBLIC/'analysis.json'); v = source.read(PUBLIC/'verification.json')
    api.base.exact(v['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    f = source.read(PUBLIC/'training_freeze.json')
    if f['complete_heads'] != 9 or not v['all9_initial_final_fitting_metrics_exact'] or not source.read(PUBLIC/'full_refit.json')['exact']:
        raise ValueError('Complete real training, exact refit and metric replay required')
    rows = [source.read(source.checked(ref)) for ref in f['fits']]
    for r in rows:
        s = api.base.read_checkpoint(source.checked(r['checkpoint']))
        if s['step'] != 2000 or not r['neural_encoders_updated']:
            raise ValueError('Real fixed-budget neural fits required')
    screen = run.previous.api.fitting_screen([r for r in rows if r['arm'] == 'separate64'])
    api.base.exact(screen, a['fitting_screen'])
    held = a['status'] == 'fresh_run'
    if held != screen['followup_readout_allowed']:
        raise ValueError('No posthoc readout selection')
    phases = {p: source.read(PUBLIC/(p+'_execution.json')) for p in ('pilot', 'train', 'evaluate', 'verify')}
    comparisons = []
    for job in sorted({r['job'] for r in rows}):
        by = {r['arm']: r['final_fitting'] for r in rows if r['job'] == job}
        for control in ('shared128', 'shared64'):
            comparisons.append(dict(job=job, treatment='separate64', control=control,
                MSE_reduction_percent=reduction(by[control]['mean_MSE'], by['separate64']['mean_MSE']),
                sign_regret_reduction_percent=[reduction(x, y) for x, y in
                    zip(by[control]['observed_sign_regret'], by['separate64']['observed_sign_regret'])]))
    summary = dict(status=a['status'], training='fresh_run', fitting=[compare(r) for r in rows],
        architecture_comparisons=comparisons, fitting_screen=screen,
        held_readout='fresh_run_exposed_development' if held else 'not_run',
        held_readout_reason=None if held else a['reason'], phases=phases,
        primary_updates=f['primary_updates'], additional_full_refit_updates=f['additional_full_refit_updates'],
        all9_initial_final_metrics_exact=True, full_refit_exact=True,
        original_rules_unchanged=True, identity_certification=False, new_H_readout=False,
        independent_confirmation=False, original24group_primary='not_estimable_not_passed',
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'result_summary.json', source.stable(summary))
    fmt = lambda x: 'NA' if x is None else f'{x:.3f}'
    pair = lambda x: ' / '.join(fmt(v) for v in x)
    lines = ['# Separate Risk And Decision Representations', '',
        'I trained three neural architectures in each of three own-fitting contexts',
        'under the unchanged 2% all/easy risk and support rules. The separated model',
        'learns risk magnitude and intervention decisions in independent encoders.',
        'Narrow and near-capacity-matched wide shared encoders are fixed controls.',
        'All nine final iterates are reported; none is selected using held outcomes.', '',
        '## Complete Fitting Results', '',
        '| Context | Architecture | Parameters | Mean MSE reduction % | Easy MSE reduction % | All/easy decision-cost reduction % |',
        '|---|---|---:|---:|---:|---:|']
    for r in summary['fitting']:
        lines.append(f"| {r['job']} | {r['arm']} | {r['parameters']} | {fmt(r['MSE_reduction_percent'])} | {fmt(r['easy_MSE_reduction_percent'])} | {pair(r['sign_regret_reduction_percent'])} |")
    lines += ['', 'Positive means improvement on fitting data versus its warm initialization.',
        'These are normalized risk-regression losses and observed-label decision costs,',
        'NOT trajectory ADE/FDE, test gains, or safety probabilities. Conservative',
        'missing-label supervision remains active in both heads. Training all/easy',
        'cost weighting is unchanged. A fall in MSE alone does not pass the screen.', '',
        '## Capacity-Controlled Comparison', '',
        '| Context | Separate64 versus | MSE reduction % | All/easy decision-cost reduction % |',
        '|---|---|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['job']} | {r['control']} | {fmt(r['MSE_reduction_percent'])} | {pair(r['sign_regret_reduction_percent'])} |")
    lines += ['', 'Wide shared has256 more parameters than separated (64,516 versus64,260).',
        'The primary comparison is against this wide control; the narrow control has',
        '32,260 parameters. Initialization is function-matched within numerical',
        'tolerance, not identical parameterization. Shared/separate gradients differ.',
        'This is not a proof of capacity-independent superiority or architectural novelty.', '',
        '## Development Readout', '', f"Prospective fitting expansion screen: **{screen['followup_readout_allowed']}**."]
    if not held:
        lines += ['**not_run:** '+a['reason']+'.',
            'No new held utility or safety result is inferred from fitting improvements.',
            'Earlier held failures remain failures. The model is not deployed.']
    else:
        lines += ['All nine models froze before the same already exposed nested-C readout.',
            '| Policy | Known floor gain % | Hard gain % | Switch % | Full risk/support | Worst easy-net % |',
            '|---|---:|---:|---:|---:|---:|']
        for name in ('readout_veto', 'constant_veto', 'shared64_veto', 'shared128_veto', 'separate64_veto',
                'shared64_risk_only_veto', 'shared128_risk_only_veto', 'separate64_risk_only_veto'):
            p = a['held_policy'][name]; safety = p['safety']
            lines.append(f"| {name} | {fmt(p['known']['mean'])} | {fmt(p['hard']['mean'])} | {fmt(p['switch_percent']['mean'])} | {safety['supported']}/{safety['required']} | {fmt(safety['worst_defined_signed_easy_net_percent'])} |")
        lines += ['', '| Contrast | Known pp | Completion lower pp | Completion upper pp |', '|---|---:|---:|---:|']
        for name, r in a['contrasts'].items():
            lines.append('| '+name+' | '+' | '.join(fmt(r[k]['mean']) for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp'))+' |')
        lines += ['', 'Completion bounds are NOT sampling CIs. Partial known ADE is not',
            'complete-roster12-step ADE. Repeated development on seven SOURCE units',
            '/17 recording views and one producer seed is not independent confirmation.',
            'Cached controls are hash-verified and their metrics freshly recomputed.']
    lines += ['', '## Computation And Replay', '',
        '18,000 primary updates, 2,000 exact full-refit updates, and600 pilot updates',
        'including repeats. Each fit uses the same2,000 query draws as its original',
        'constant control. All9 initial/final complete-fitting metrics replay exactly.',
        'All3 arms pass50+50 versus100 actual-update resume and prediction equality.',
        'The first separate64 full fit repeats exactly. A budget cap is not convergence.', '',
        '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for name, r in phases.items():
        lines.append(f"| {name} | {r['pid']} | {r['seconds']:.2f} | {r['peak_RSS_bytes']} |")
    lines += ['', 'A completed evaluate command can only record not_run; it is not evidence',
        'that held predictions ran. Real arm64 PyTorch CPU4/inter-op1/workers0 was',
        'used, not NumPy fallback. Checkpoints remain private and hash-bound.', '',
        'Original24-group primary remains NA, not passed. Empty controls/missing',
        'support remain NA; no identity certification or new H/reserved readout.',
        'This is an intervention-risk model, not newly trained trajectory dynamics.',
        'Native obs8/pred12 image coordinates; no metric/seconds/true3D/foundation',
        'or human-gold claim. Stage5C and SMC remain disabled.', '',
        '[Protocol](protocol.md) | [Numeric summary](result_summary.json) | [Analysis](analysis.json)',
        '[Interpretation](result_interpretation.md) | [Recovery guide](operation_zh.md)', '']
    (PUBLIC/'README.md').write_text('\n'.join(lines))
    (PUBLIC/'reproducibility.md').write_text('# Reproducibility\n\n```sh\n'+
        '\n'.join('.venv-pytorch/bin/python scripts/run_m3w_moving_decoupled_risk.py '+p
            for p in ('register', 'pilot', 'train', 'evaluate', 'verify'))+
        '\n.venv-pytorch/bin/python scripts/report_m3w_moving_decoupled_risk.py\n```\n\n'
        'Completed phases refuse overwrite. Only resume an interrupted unchanged\n'
        'phase with --resume. Exact replay uses existing checked artifacts, not\n'
        'new data independence. This is a one-producer-seed model diagnostic.\n')
    original_safety = a['held_policy']['separate64_veto']['safety']['full_original_risk_pass'] if held else None
    (PUBLIC/'gates.md').write_text('\n'.join([
        '# Evidence Gates', '', '| Check | Result | Scope |', '|---|---|---|',
        '| Nine real fits completed | pass | Engineering/training, not scientific success |',
        '| Exact pilot/full-refit/metric replay | pass | Reproducibility on existing data |',
        '| Original thresholds and data roles retained | pass | No identity certificate or new H |',
        f"| Prespecified fitting expansion screen | {screen['followup_readout_allowed']} | Internal diagnostic, not safety |",
        f"| Original full risk/support on exposed readout | {original_safety if held else 'not_run'} | No missing-support exemption |",
        '| Original24-group primary | NA, not passed | Reference-support limitation remains |',
        '| Independent confirmation | not_run | Reserved data remain closed |',
        '| Deployment upgrade | false | No promotion from fitting evidence |',
        '| Submission ready | false | Independent scientific evidence incomplete |',
        '| Stage5C / SMC execution | false / false | Prohibited, not scientific achievements |', '',
        'No aggregate gate count combines engineering progress with scientific success.', '']))
    (PUBLIC/'model_card.md').write_text('\n'.join([
        '# Model And Data Card', '',
        'Purpose: a fixed intervention-risk learning diagnostic. The frozen trajectory',
        'producer is not retrained. These heads do not generate new trajectories.',
        'Three own-B contexts, three prespecified architectures, one producer seed.',
        'Original492 causal features: past/history/neighbor and frozen-candidate',
        'rollout context with an explicit64-dimensional motion basis. Previous latent',
        'input columns remain zeroed as in the parent experiment; no new JEPA claim.',
        'Complete current-query pooling; no future endpoint/validity mask at inference.',
        'Preprocessing, target and proposal/observed-mask hashes are bound to each',
        'own-fitting checkpoint. Outcomes and their masks are used only for losses',
        'and evaluation. Old source/teacher lineage restrictions are inherited.', '',
        'Outputs: all/easy signed risk regression and cost-weighted decision logits.',
        'Both signs must accept, plus original support requirements. Neither score',
        'is a certified upper risk bound. Decision magnitudes are not probabilities.',
        'Shared64/shared128/separate64 have32,260/64,516/64,260 parameters.',
        'AdamW .0001, fixed2,000 updates, original query sampler, no V early stopping.',
        'Both heads retain conservative unknown-outcome supervision.', '',
        'Evidence: all9 real fits and exact initial/final metric replay complete.',
        f"Additional held evaluation: {'fresh_run on exposed development only' if held else 'not_run after failed fitting screen'}.",
        'No independent test, deployment upgrade or submission-ready claim.',
        'Native obs8/pred12 image coordinates; no metric/seconds/true3D/foundation',
        'or human-gold status. Original24-group primary remains NA, not passed.',
        'Private checkpoints and caches are not in Git. Public hashes alone do not',
        'make training reproducible without the permitted local data assets.',
        'Stage5C and SMC remain disabled.', '']))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
