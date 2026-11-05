"""Summarize verified fixed-loss experiments, never choose a model from readout."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_joint_decision as run
from scripts.report_m3w_moving_decoupled_risk import reduction, compare

source, api, PUBLIC = run.source, run.api, run.PUBLIC


def verify_roster(rows, jobs):
    required = {(j['key'], arm) for j in jobs for arm in api.ARMS}
    actual = [(r['job'], r['arm']) for r in rows]
    if (len(jobs) != 3 or len(rows) != 6 or len(set(actual)) != 6 or set(actual) != required
            or any(r['step'] != 2000 or r['parameters'] != 64260
                or not r['both_neural_encoders_updated'] for r in rows)):
        raise ValueError('All six unique registered complete neural fits required')


def main():
    a = source.read(PUBLIC/'analysis.json'); verified = source.read(PUBLIC/'verification.json')
    api.base.exact(verified['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    freeze = source.read(PUBLIC/'training_freeze.json')
    if (freeze['complete_heads'] != 6 or not freeze['all3_controls_reproduce_previous']
            or not verified['all6_initial_final_fitting_metrics_exact']
            or not verified['all6_joint_attributions_exact'] or not source.read(PUBLIC/'full_refit.json')['exact']):
        raise ValueError('Completed training, exact controls/refit/attribution required')
    rows = [source.read(source.checked(ref)) for ref in freeze['fits']]
    verify_roster(rows, source.read(run.REG)['jobs'])
    for r in rows:
        state = api.base.read_checkpoint(source.checked(r['checkpoint']))
        if state['step'] != 2000 or not r['both_neural_encoders_updated']:
            raise ValueError('Real full-budget updates required')
    screen = run.previous.previous.previous.api.fitting_screen([r for r in rows if r['arm'] == 'joint'])
    api.base.exact(screen, a['fitting_screen'])
    held = a['status'] == 'fresh_run'
    if held != screen['followup_readout_allowed']:
        raise ValueError('Prespecified screen determines held readout')
    pairs, source_pairs = [], []
    for job in sorted({r['job'] for r in rows}):
        arms = {r['arm']: r for r in rows if r['job'] == job}
        left, right = arms['per_channel'], arms['joint']
        la, ra = left['joint_attribution']['total'], right['joint_attribution']['total']
        pairs.append(dict(job=job,
            MSE_reduction_percent=reduction(left['final_fitting']['mean_MSE'], right['final_fitting']['mean_MSE']),
            channel_cost_reduction_percent=[reduction(x, y) for x, y in zip(
                left['final_fitting']['observed_sign_regret'], right['final_fitting']['observed_sign_regret'])],
            joint_cost_reduction_percent=reduction(la['total_joint_cost'], ra['total_joint_cost']),
            false_reject_cost_reduction_percent=reduction(la['false_reject_cost'], ra['false_reject_cost']),
            false_safe_cost_reduction_percent=reduction(la['false_safe_cost'], ra['false_safe_cost']),
            zero_only_reject_cost_reduction_percent=reduction(la['false_reject_partition']['zero_only']['cost'],
                ra['false_reject_partition']['zero_only']['cost'])))
        local_a, local_b = left['joint_attribution']['by_source'], right['joint_attribution']['by_source']
        if set(local_a) != set(local_b):
            raise ValueError('Same fitting sources required for attribution')
        for site in sorted(local_a):
            x, y = local_a[site], local_b[site]
            source_pairs.append(dict(job=job, source=site,
                control_joint_cost=x['total_joint_cost'], joint_joint_cost=y['total_joint_cost'],
                joint_cost_change=y['total_joint_cost']-x['total_joint_cost'],
                false_safe_cost_change=y['false_safe_cost']-x['false_safe_cost'],
                false_reject_cost_change=y['false_reject_cost']-x['false_reject_cost']))
    phases = {p: source.read(PUBLIC/(p+'_execution.json')) for p in ('pilot', 'train', 'evaluate', 'verify')}
    summary = dict(status=a['status'], fitting=[compare(r) for r in rows], matched_contrasts=pairs,
        source_contrasts=source_pairs,
        fitting_screen=screen, phases=phases, training='fresh_run_complete',
        primary_updates=12000, exact_refit_updates=2000, pilot_updates=400,
        all3_controls_reproduce_previous_exactly=True, full_refit_exact=True,
        all6_metrics_and_joint_attributions_exact=True,
        held_readout='fresh_run_exposed_development' if held else 'not_run',
        held_readout_reason=None if held else a['reason'],
        original_rules_unchanged=True, identity_certification=False, new_H=False,
        independent_confirmation=False, original24group_primary='not_estimable_not_passed',
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'result_summary.json', source.stable(summary))
    f = lambda x: 'NA' if x is None else f'{x:.3f}'
    pair = lambda x: ' / '.join(f(v) for v in x)
    text = ['# Learning The Complete Intervention Decision', '',
        'I tested one loss repair after a fitting-only audit found that zero-target',
        'channels can block useful proposals despite receiving no signed-cost',
        'gradient. The treatment trains the complete four-output conjunction.',
        'Risk MSE, the original decision objective, conservative missing-label',
        'supervision and the original2% risk/easy-net/support rules are unchanged.', '',
        '## Controlled Experiment', '',
        'Six real neural fits compare per-channel and joint auxiliary losses in',
        'three own-B contexts. All use64,260 parameters,492 causal inputs, identical',
        'warm starts, query draws and2,000 updates. The three fresh controls exactly',
        'reproduce the preceding composed models, optimizers and predictions.',
        'The first full joint fit and both pilot resume paths reproduce exactly.', '',
        '| Context | Loss | MSE reduction from warm % | All/easy sign-cost reduction from warm % |',
        '|---|---|---:|---:|']
    for r in summary['fitting']:
        text.append(f"| {r['job']} | {r['arm']} | {f(r['MSE_reduction_percent'])} | {pair(r['sign_regret_reduction_percent'])} |")
    text += ['', 'These are fitting diagnostics, not trajectory ADE/FDE or test gains.', '',
        '| Context | Matched MSE reduction % | All/easy cost reduction % | Joint cost reduction % | False-reject reduction % | False-safe reduction % | Zero-only reject reduction % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in pairs:
        text.append(f"| {r['job']} | {f(r['MSE_reduction_percent'])} | {pair(r['channel_cost_reduction_percent'])} | " +
            ' | '.join(f(r[k]) for k in ('joint_cost_reduction_percent', 'false_reject_cost_reduction_percent',
                'false_safe_cost_reduction_percent', 'zero_only_reject_cost_reduction_percent'))+' |')
    passed = sum(c['passed'] for c in screen['contexts'])
    text += ['', f'Original internal expansion screen: **{passed}/3**, overall **{screen["followup_readout_allowed"]}**.',
        'Each treatment context must improve risk MSE at least1% from warm and',
        'not worsen either channel sign cost. This is not a scientific safety gate.',
        'Joint attribution is diagnostic; it cannot replace a failed screen.', '',
        '## Joint Error Attribution', '',
        '| Context | Loss | Target-safe proposals | False rejections | False-safe proposals | Zero-only rejection cost | Nonzero-only cost | Both cost |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        m = r['joint_attribution']['total']; p = m['false_reject_partition']
        text.append(f"| {r['job']} | {r['arm']} | {m['true_joint_safe_proposals']} | {m['false_reject_count']} | "
            f"{m['false_safe_count']} | {p['zero_only']['cost']:.8f} | {p['nonzero_only']['cost']:.8f} | {p['both']['cost']:.8f} |")
    text += ['', 'Zero-only/nonzero-only/both are disjoint. Per-output blocker counts',
        'overlap and cannot be added as independent samples. Target-safe means',
        'observed-label normalized sign only, not a deployment safety certificate.',
        'Per-source attribution is retained in every fitting receipt.', '',
        '| Context | Fitting source | Joint cost change | False-safe cost change | False-reject cost change |',
        '|---|---|---:|---:|---:|']
    for r in source_pairs:
        text.append(f"| {r['job']} | {r['source']} | " + ' | '.join(f"{r[k]:+.8f}" for k in
            ('joint_cost_change', 'false_safe_cost_change', 'false_reject_cost_change'))+' |')
    text += ['', 'Positive changes above are worse. These are absolute contributions',
        'under the same original query weighting, not source-standardized effects',
        'or independent-sample estimates. Negative slices are retained.', '', '## Held Development', '']
    if not held:
        text += ['**not_run:** '+a['reason']+'.',
            'The evaluate command records this decision; it does not evaluate new',
            'held predictions. No held improvement is inferred from fitting.']
    else:
        text += ['Seven already exposed SOURCE units /17 recording views, one producer',
            'seed. Every model/action freezes before outcomes are accessed.', '',
            '| Policy | Known ADE gain % | Hard gain % | Switch % | Worst defined easy-net % | Full original risk/support |',
            '|---|---:|---:|---:|---:|---:|']
        for name in ('constant_veto', 'per_channel_veto', 'joint_veto', 'joint_risk_only_veto'):
            v = a['held_policy'][name]; s = v['safety']
            text.append(f"| {name} | {f(v['known']['mean'])} | {f(v['hard']['mean'])} | {f(v['switch_percent']['mean'])} | "
                f"{f(s['worst_defined_signed_easy_net_percent'])} | {s['supported']}/{s['required']} |")
        text += ['', '| Contrast | Known pp | Completion lower pp | Completion upper pp |', '|---|---:|---:|---:|']
        for name, row in a['contrasts'].items():
            text.append('| '+name+' | '+' | '.join(f(row[k]['mean']) for k in ('known_pp','completion_lower_pp','completion_upper_pp'))+' |')
        text += ['', 'Completion bounds are not sampling CIs. Known ADE uses available',
            'outcomes, not complete-roster12-step accuracy. Repeated exposed',
            'development is not independent confirmation or multi-seed evidence.']
    text += ['', '## Execution And Evidence', '',
        '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for key, r in phases.items():
        text.append(f"| {key} | {r['pid']} | {r['seconds']:.2f} | {r['peak_RSS_bytes']} |")
    text += ['', '12,000 primary updates,2,000 exact full-refit updates,400 pilot updates.',
        'Real arm64 PyTorch CPU4/inter-op1/workers0, not NumPy fallback. All six',
        'initial/final metrics and joint attributions replay exactly. Models,',
        'optimizers, sampler and RNG are checkpointed every100 completed updates.', '',
        'Parent data/schema/warm assets: cached_verified. Training and metrics:',
        'fresh_run. New reserved/H confirmation: not_run. Private arrays and weights',
        'remain excluded from Git. Hashes alone are not a self-contained benchmark.', '',
        'No deployment or submission-ready claim. Original24-group primary remains',
        'NA, not passed; empty controls and missing support remain NA. No identity',
        'certification, threshold change or reserved Moving/DUT/DroneCrowd/H reading.',
        'Neural intervention-risk training, not new JEPA or forecasting dynamics.',
        'Native obs8/pred12 image coordinates, not metric/seconds/true3D/foundation/',
        'human-gold evidence. Stage5C and SMC disabled.', '',
        '[Protocol](protocol.md) | [Numeric summary](result_summary.json) |',
        '[Failure analysis](failure_analysis.md) | [Recovery guide](operation_zh.md)', '']
    (PUBLIC/'README.md').write_text('\n'.join(text))
    safety = a['held_policy']['joint_veto']['safety']['full_original_risk_pass'] if held else 'not_run'
    (PUBLIC/'gates.md').write_text('\n'.join([
        '# Joint-Decision Evidence Gates', '', '| Gate | Result | Meaning |', '|---|---|---|',
        '| Six actual fits | pass | Training completed, not scientific success |',
        '| Matched controls, resume and full refit | pass | Exact replay |',
        '| Six fitting metrics and attribution replay | pass | Same-data reproducibility |',
        '| Original rules and roles preserved | pass | No identity exemption or new H |',
        f'| Internal fitting screen | {passed}/3; {screen["followup_readout_allowed"]} | Fixed diagnostic, not redefined |',
        f'| Full original held risk/support | {safety} | Scientific safety criterion |',
        '| Original24-group primary | NA, not passed | Reference-support limitation |',
        '| Independent confirmation | not_run | Reserved roles closed |',
        '| Deployment / submission ready | false / false | Not inferred from training |',
        '| Stage5C / SMC execution | false / false | Disabled, not scientific successes |', '',
        'No gate total mixes execution with scientific success.', '']))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
