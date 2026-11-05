"""Report the frozen matched loss contrast without selecting models or thresholds."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_composed_risk as run
from scripts.report_m3w_moving_decoupled_risk import reduction, compare

source, api, PUBLIC = run.source, run.api, run.PUBLIC


def main():
    analysis = source.read(PUBLIC/'analysis.json')
    verification = source.read(PUBLIC/'verification.json')
    freeze = source.read(PUBLIC/'training_freeze.json')
    api.base.exact(verification['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    attribution = source.read(PUBLIC/'fitting_error_attribution.json')
    replay = source.read(PUBLIC/'fitting_error_attribution_replay.json')
    api.base.exact(replay['attribution_sha256'], source.sha(PUBLIC/'fitting_error_attribution.json'))
    if (freeze['complete_heads'] != 6 or not freeze['all3_controls_reproduce_previous'] or
            not verification['all6_initial_final_fitting_metrics_exact'] or
            not replay['all6_decompositions_exact'] or not source.read(PUBLIC/'full_refit.json')['exact']):
        raise ValueError('Complete real fits, controls and exact replay required')
    rows = [source.read(source.checked(ref)) for ref in freeze['fits']]
    for row in rows:
        state = api.base.read_checkpoint(source.checked(row['checkpoint']))
        if state['step'] != 2000 or not row['both_neural_encoders_updated']:
            raise ValueError('Both encoders must actually train through the fixed budget')
    screen = run.previous.previous.api.fitting_screen([r for r in rows if r['arm'] == 'composed'])
    api.base.exact(screen, analysis['fitting_screen'])
    held = analysis['status'] == 'fresh_run'
    if held != screen['followup_readout_allowed']:
        raise ValueError('Frozen fitting screen determines readout, not posthoc selection')
    phases = {p: source.read(PUBLIC/(p+'_execution.json')) for p in
        ('pilot', 'train', 'evaluate', 'verify', 'attribution', 'attribution_replay')}
    contrasts = []
    for job in sorted({r['job'] for r in rows}):
        by_arm = {r['arm']: r['final_fitting'] for r in rows if r['job'] == job}
        left, right = by_arm['independent'], by_arm['composed']
        contrasts.append(dict(job=job, treatment='composed', control='independent',
            MSE_reduction_percent=reduction(left['mean_MSE'], right['mean_MSE']),
            sign_regret_reduction_percent=[reduction(x, y) for x, y in
                zip(left['observed_sign_regret'], right['observed_sign_regret'])]))
    summary = dict(status=analysis['status'], training='fresh_run_complete', fitting=[compare(r) for r in rows],
        matched_loss_contrasts=contrasts, fitting_screen=screen,
        held_readout='fresh_run_exposed_development' if held else 'not_run',
        held_readout_reason=None if held else analysis['reason'], phases=phases,
        primary_updates=12000, additional_full_refit_updates=2000, all3_controls_reproduced_exactly=True,
        all6_initial_final_metrics_exact=True, all6_attributions_exact=True, full_refit_exact=True,
        original_rules_unchanged=True, identity_certification=False, new_H_readout=False,
        independent_confirmation=False, original24group_primary='not_estimable_not_passed',
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'result_summary.json', source.stable(summary))
    fmt = lambda value: 'NA' if value is None else f'{value:.3f}'
    pair = lambda values: ' / '.join(fmt(v) for v in values)
    lines = ['# Learning The Composed Intervention Decision', '',
        'I tested whether directly training the conjunction of two neural decisions',
        'corrects the false rejections seen with independently trained heads. Both',
        'arms retain the original risk-regression and conservative missing-label',
        'losses. The treatment adds an observed-label cost-sensitive loss on their',
        'per-channel maximum. The 2% risk, easy-net and support rules are unchanged.', '',
        '## Fixed Matched Comparison', '',
        'Six actual fits use the same three fitting contexts, 64,260 parameters,',
        '492 causal inputs, warm initialization, query draws and 2,000 updates.',
        'All three fresh independent controls exactly reproduce the previous',
        'separated models, optimizer states and predictions. No checkpoint or',
        'threshold is selected from held outcomes. The final treatment is repeated',
        'exactly for the first context. A finite update budget is not convergence.', '',
        '| Context | Loss | Risk MSE reduction % | All/easy decision-cost reduction % |',
        '|---|---|---:|---:|']
    for row in summary['fitting']:
        lines.append(f"| {row['job']} | {row['arm']} | {fmt(row['MSE_reduction_percent'])} | {pair(row['sign_regret_reduction_percent'])} |")
    lines += ['', 'Positive means improvement on fitting data versus its warm model.',
        'These are risk-regression losses and observed sign costs, NOT trajectory',
        'ADE/FDE or test improvements. Unknown labels keep their original bounds.', '',
        '| Context | Composed versus independent: MSE reduction % | All/easy cost reduction % |',
        '|---|---:|---:|']
    for row in contrasts:
        lines.append(f"| {row['job']} | {fmt(row['MSE_reduction_percent'])} | {pair(row['sign_regret_reduction_percent'])} |")
    passed = sum(c['passed'] for c in screen['contexts'])
    lines += ['', f'Internal fitting expansion screen: **{passed}/3 contexts**, overall **{screen["followup_readout_allowed"]}**.',
        'All three must reduce mean risk MSE by at least1% and not increase either',
        'observed combined sign cost. This is an internal readout screen, not the',
        'scientific safety gate. The matched loss contrast is also reported above.', '',
        '## Held Development Readout', '']
    if not held:
        lines += ['**not_run:** '+analysis['reason']+'.',
            'A completed evaluate command only records this decision; it does not',
            'mean new held predictions were produced. No held gain is inferred.']
    else:
        lines += ['Both arms and risk-only diagnostics were frozen before evaluation',
            'on the same seven exposed SOURCE units /17 recording views.', '',
            '| Policy | Known ADE gain % | Hard gain % | Switch % | Full original risk/support | Worst easy-net % |',
            '|---|---:|---:|---:|---:|---:|']
        for name in ('constant_veto', 'independent_veto', 'composed_veto',
                'independent_risk_only_veto', 'composed_risk_only_veto'):
            policy = analysis['held_policy'][name]; safety = policy['safety']
            lines.append(f"| {name} | {fmt(policy['known']['mean'])} | {fmt(policy['hard']['mean'])} | "
                f"{fmt(policy['switch_percent']['mean'])} | {safety['supported']}/{safety['required']} | "
                f"{fmt(safety['worst_defined_signed_easy_net_percent'])} |")
        lines += ['', '| Contrast | Known pp | Completion lower pp | Completion upper pp |',
            '|---|---:|---:|---:|']
        for name, contrast in analysis['contrasts'].items():
            lines.append('| '+name+' | '+' | '.join(fmt(contrast[k]['mean']) for k in
                ('known_pp', 'completion_lower_pp', 'completion_upper_pp'))+' |')
        lines += ['', 'Completion bounds are not sampling confidence intervals. Partial',
            'known ADE is not complete-roster12-step ADE. Cached controls are',
            'hash-verified with exact fresh metric recomputation. Repeated exposed',
            'development is not independent confirmation or a three-seed primary.']
    lines += ['', '## Error Attribution', '',
        '| Context | Loss | All false-safe cost reduction % | All false-reject cost reduction % | Easy false-safe cost reduction % | Easy false-reject cost reduction % |',
        '|---|---|---:|---:|---:|---:|']
    for row in attribution['rows']:
        initial, final = row['total']['initial_combined'], row['total']['final_combined']
        values = [reduction(initial[k][channel], final[k][channel])
            for channel in range(2) for k in ('false_safe_cost', 'false_reject_cost')]
        lines.append('| '+row['job']+' | '+row['arm']+' | '+' | '.join(fmt(v) for v in values)+' |')
    lines += ['', 'All per-source decompositions are in fitting_error_attribution.json.',
        'These analyses diagnose the fixed models; they do not select a stand-alone',
        'branch or erase failures. The maximum trains per-channel sign conjunction,',
        'not the full population risk ratio or reference support. Its magnitudes',
        'are not calibrated probabilities or certified upper risk bounds.', '',
        '## Runtime And Evidence Scope', '',
        '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for name, phase in phases.items():
        lines.append(f"| {name} | {phase['pid']} | {phase['seconds']:.2f} | {phase['peak_RSS_bytes']} |")
    lines += ['', '12,000 primary updates +2,000 exact full-refit updates +400 pilot updates.',
        'Real native arm64 PyTorch CPU4/inter-op1/workers0, not NumPy fallback.',
        'All six initial/final fitting metrics and error decompositions replay exactly.',
        'Private model/data files remain excluded from Git; only light receipts are public.', '',
        '| Evidence | Origin | Meaning |', '|---|---|---|',
        '| Data/schema/partitions/warm starts | cached_verified | Inherited bindings checked, not new collection |',
        '| Six model fits and fitting attribution | fresh_run | Real training and fresh computation |',
        '| Independent control match | fresh_run plus cached_verified | Fresh fit compared with frozen previous artifact |',
        f'| Additional nested-C development | {summary["held_readout"]} | No independent-test claim |',
        '| Reserved/H confirmation | not_run | Remains closed |', '',
        'No deployment upgrade or submission-ready claim. The original24-group',
        'primary remains NA, not passed; empty controls and missing support stay NA.',
        'No identity certification or new H/reserved Moving/DUT/DroneCrowd reading.',
        'This trains intervention-risk heads, not new JEPA or trajectory dynamics.',
        'Native obs8/pred12 image coordinates; no metric, seconds, true3D, foundation',
        'or human-gold claim. Stage5C and SMC remain disabled.', '',
        '[Protocol](protocol.md) | [Numeric summary](result_summary.json)',
        '[Interpretation](result_interpretation.md) | [Recovery](operation_zh.md)', '']
    (PUBLIC/'README.md').write_text('\n'.join(lines))
    safety = analysis['held_policy']['composed_veto']['safety']['full_original_risk_pass'] if held else 'not_run'
    (PUBLIC/'gates.md').write_text('\n'.join([
        '# Evidence Gates', '', '| Check | Result | Scope |', '|---|---|---|',
        '| Six real fits | pass | Actual optimizer updates, not scientific success |',
        '| Three controls exactly reproduce previous models | pass | Isolated loss change |',
        '| Resume, full refit, metrics and attribution replay | pass | Existing-data reproducibility |',
        '| Original labels, thresholds and roles retained | pass | No identity exemption or new H |',
        f'| Fitting expansion screen | {passed}/3, {screen["followup_readout_allowed"]} | Internal diagnostic |',
        f'| Full original held risk/support | {safety} | Scientific criterion, unchanged |',
        '| Original24-group primary | NA, not passed | Reference-support limitation |',
        '| Independent confirmation | not_run | Reserved sources remain closed |',
        '| Deployment upgrade / submission ready | false / false | Not inferred from fitting |',
        '| Stage5C / SMC execution | false / false | Prohibited, not scientific achievements |', '',
        'No gate total mixes engineering completion with scientific validation.', '']))
    (PUBLIC/'model_card.md').write_text('\n'.join([
        '# Composed Decision Model Card', '',
        'Purpose: cost-aware intervention-risk learning. The trajectory producer is frozen.',
        'Six fixed models: two loss arms in three own-B contexts, one producer seed.',
        'Architecture: separate64 risk and decision encoders,64,260 parameters each model.',
        '492 causal inputs include history, neighbors and frozen candidate rollouts.',
        'As in the parent, previous latent columns are zeroed; no new JEPA contribution.',
        'Current-query pooling uses no future endpoint or label-validity feature.',
        'Own-fitting preprocessing, input/target/proposal/observed-mask hashes are bound.',
        'Future outcomes and validity are supervision/evaluation only.', '',
        'Outputs: all/easy signed risk estimates and cost-sensitive decision logits.',
        'Both signs must accept and original support must hold. Neither is a calibrated',
        'risk guarantee. The new max-based surrogate trains their per-channel conjunction.',
        'Original risk MSE, decision loss and conservative unknown supervision remain.',
        'Fixed AdamW .0001, clip5,2,000 updates; no validation checkpoint/threshold search.',
        'Fresh independent controls match preceding models exactly. Both encoders update.', '',
        f'Additional held evaluation: {summary["held_readout"]}.',
        'No independent confirmation, deployment upgrade or submission-ready claim.',
        'Native obs8/pred12 image coordinates, not metric or seconds. Not true3D,',
        'foundation or human-gold evidence. Stage5C and SMC remain disabled.',
        'Private checkpoints/caches are not in Git. Permitted local assets are needed',
        'to replay; hashes alone do not constitute a self-contained public benchmark.', '']))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
