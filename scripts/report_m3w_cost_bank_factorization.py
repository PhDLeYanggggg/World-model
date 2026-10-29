"""Report every fixed fitting-only bank control without selecting a deployment."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import diagnose_m3w_cost_bank_factorization as run


def number(value):return 'not_estimable' if value is None else f'{value:+.6f}'


def interval(value):
    return 'not_estimable' if value['CI95'] is None else '['+', '.join(number(x) for x in value['CI95'])+']'


def build():
    value=json.loads((run.PUBLIC/'summary.json').read_text())
    check=json.loads((run.PUBLIC/'verification.json').read_text())
    complete=json.loads((run.PUBLIC/'complete.json').read_text())
    if not check['exact_replay'] or check['groups']!=144 or check['summary_sha256']!=run.sha(run.PUBLIC/'summary.json'):
        raise ValueError('Full separate exact inference replay required')
    gates=dict(all144fitting_groups_complete=True,original_policies_and_neutral_replay_exact=True,
        held_scored=False,new_optimizer_updates=0,new_trees=0,
        independent_confirmation=False,deployment_promoted=False,submission_ready=False,
        stage5c_executed=False,smc_enabled=False)
    for family,stats in value['families'].items():
        p=stats['safety']['factorized'];ci=stats['contrasts']['factorized']['full_known_pp']
        gates[family+'_factorized_fitting_full_risk_supported']=p['supported_record_views']==p['required_record_views']
        gates[family+'_factorized_fitting_gain_CI_positive']=ci['CI95'] is not None and ci['CI95'][0]>0
    return dict(summary_sha256=run.sha(run.PUBLIC/'summary.json'),verification_sha256=run.sha(run.PUBLIC/'verification.json'),
        result_source='fresh_report_verified_fitting_inference_cached_verified_models',summary=value,gates=gates,
        execution={key:complete[key] for key in ('pid','seconds','peak_RSS_bytes')},
        replay=json.loads((run.PRIVATE/'verification_runtime.json').read_text()))


def render(value):
    summary=value['summary'];execution=value['execution'];replay=value['replay']
    lines=['# Frozen Cost-Bank Factorization: Fitting-Only Evidence','',
        '## Material Passport','',
        'Code experiment / fixed inference control. Status: real run and separate exact replay complete.',
        'Fresh: current fitting inference, controls and metrics. Cached_verified: frozen models,',
        'causal feature packets and prior negative reports. Not_run: new training, held or',
        'independent evaluation, world-dynamics learning and deployment.','',
        '## Execution','',
        f"Run PID {execution['pid']}, exit 0, completes 144 groups in {execution['seconds']:.6f}s; peak RSS {execution['peak_RSS_bytes']:,} B.",
        f"Separate PID {replay['pid']}, exit 0, exact replay takes {replay['seconds']:.6f}s.",
        'All original policy choice hashes and complete metrics match the preceding frozen',
        'fitting readout. Mixed-with-itself neutral costs, choices and ranks are exact.',
        'No optimizer step or new tree. Three head seeds do not create three forecasting pipelines.','',
        '## What Stops The Neural Candidate?','',
        '| Cost-estimator family | Eligible neural query views | Mixed neural gate passes | Mixed neural choices | Lost to physical competition | Dedicated neural expert passes | Dedicated passes rejected by mixed gate |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for family,stats in summary['families'].items():
        g=stats['gate6']
        lines.append(f"| {family} | {g['mixed']['eligible']} | {g['mixed']['passing']} | "
            f"{g['mixed_gate_pass_and_neural_selected']} | {g['mixed_gate_pass_physical_competition_lost']} | "
            f"{g['dedicated']['passing']} | {g['neural_expert_pass_mixed_gate_reject']} |")
    lines.extend(['','Among dedicated-expert passes rejected by the mixed estimator, overlapping causes:','',
        '| Family | Nonpositive gain | Zero reference | Zero easy-reference | All-harm gate | Easy-harm gate |',
        '|---|---:|---:|---:|---:|---:|'])
    for family,stats in summary['families'].items():
        row=stats['gate6']['suppression_components']
        lines.append('| '+family+' | '+' | '.join(str(row[k]) for k in run.api.FAILURES)+' |')
    lines.extend(['','Counts overlap; do not sum them as exclusive causes. Fixed-order exclusive',
        'counts and full co-occurrence bit patterns are retained in summary.json. Gate',
        'diagnostics use predicted costs and causal availability only, never future labels.','',
        '## Fixed Policy Results','',
        '| Family | Policy | Known ADE gain vs floor (%) | Known FDE gain (%) | Required recording risk support | Undefined easy views | Selected (unknown) |',
        '|---|---|---:|---:|---:|---:|---:|'])
    names=('physical_neural','mixed_neural_restricted','physical_only','neural_only','factorized',
        'suppressed_neural_expert','competition_lost','neutral','floor')
    for family,stats in summary['families'].items():
        for name in names:
            s=stats['safety'][name]
            lines.append(f"| {family} | {name} | {number(stats['accuracy'][name]['point'])} | "
                f"{number(stats['FDE'][name]['point'])} | {s['supported_record_views']}/{s['required_record_views']} | "
                f"{s['undefined_easy_record_views']} | {s['selected']} ({s['selected_unknown']}) |")
    lines.extend(['','Factorization uses two estimators: 256 trees instead of 128, or 52,618 neural cost',
        'parameters instead of 26,309. It changes learned costs, capacity and inference work,',
        'not just competition. Forecasters, paths, normalization and the 2% gate remain fixed.','',
        '## Paired Factorized Minus Mixed','',
        '| Family | Full known utility (pp), 95% CI | Full completion-lower (pp), 95% CI | Matched known utility (pp), 95% CI |',
        '|---|---|---|---|'])
    for family,stats in summary['families'].items():
        c=stats['contrasts']['factorized']
        lines.append('| '+family+' | '+' | '.join(number(c[k]['point'])+', '+interval(c[k])
            for k in ('full_known_pp','full_lower_pp','matched_known_pp'))+' |')
    lines.extend(['','These are 3,000 locality-resample fitting-resubstitution intervals after context',
        'and controller-seed averaging, not held generalization or independent confirmation.',
        'Completion utility is normalized by fixed known floor-error mass, not a fully',
        'observed population-error denominator. Count matching is utility-only. All unknown',
        'outcomes, required recordings and empty risk estimates remain.','',
        '## Interpretation Boundary','',
        'For the same mixed estimator, neural gate passes that lose to physical actions',
        'identify competition exactly. Differences between the dedicated neural expert',
        'and mixed estimator demonstrate bank-dependent learned decisions, not a unique',
        'causal attribution to class imbalance, architecture, objective or data sufficiency.',
        'Even a positive fitting effect does not promote deployment or justify a new held',
        'threshold search. Earlier failed selected-set recalibration remains negative.','',
        'The experiment uses EuropeanSquares SOURCE detector-track midpoint pixels,',
        'obs8/pred12/stride12 raw frames, not SDD human gold, metric/seconds, true 3D or',
        'foundation evidence. No future endpoint, central velocity, test goals or future',
        'support masks enter inference. Independent roles and DroneCrowd remain closed.',
        'No Stage5C execution or SMC. Submission readiness remains false.',''])
    return '\n'.join(lines)


if __name__=='__main__':
    value=build();run.once(run.PUBLIC/'evidence.json',value);run.once(run.PUBLIC/'gates.json',value['gates'])
    path=run.PUBLIC/'readout.md';text=render(value)
    if path.exists() and path.read_text()!=text:raise ValueError('Frozen report changed')
    path.write_text(text);print(json.dumps(value['gates']))
