"""Audit actual fixed-priority fits; report all original risk failures and NA."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_admission_priority as run
from scripts.diagnose_m3w_moving_admission_loss import decompose
from scripts.report_m3w_moving_direct_tail_risk import number, pool_reliability
from scripts.report_m3w_moving_readout_refit import pool_flow
from scripts.report_m3w_moving_harm_refinement import text_once

source, api, PUBLIC = run.source, run.api, run.PUBLIC
NAMES = ('mean_veto', 'readout_veto', 'kinematic_veto', 'continued_mse_veto', 'false_safe_priority_veto')


def table(policies):
    lines = ['| Policy | Known floor gain % | Hard gain % | Switch % | Full risk/support | All/easy violations | Undefined selected easy |',
        '|---|---:|---:|---:|---:|---|---:|']
    for name in NAMES:
        r = policies[name]; s = r['safety']
        lines.append(f"| {name} | {number(r['known']['mean'])} | {number(r['hard']['mean'])} | "
            f"{number(r['switch_percent']['mean'])} | {s['supported']}/{s['required']} | "
            f"{s['known_all_violations']}/{s['known_easy_violations']} | {s['undefined_selected_easy']} |")
    return '\n'.join(lines)


def audit():
    freeze = source.read(PUBLIC/'training_freeze.json'); states, fits = {}, []
    if freeze['complete_heads'] != 6 or freeze['primary_updates'] != 12000 or not freeze['all_before_validation']:
        raise ValueError('All six complete frozen real fits required')
    for ref in freeze['fits']:
        r = source.read(source.checked(ref)); s = api.base.read_checkpoint(source.checked(r['checkpoint']))
        warm = api.base.read_checkpoint(source.checked(r['warm_checkpoint']))
        if (s['step'] != 2000 or s['parameters'] != 32130 or s['risk_budget'] != .02
                or s['priority_multiplier'] != 10 or s['objective_arm'] != r['arm']):
            raise ValueError('Actual fixed training budget and objective required')
        api.base.exact(s['initial_model'], warm['model'])
        if any(int(v['step']) != 2000 for v in s['optimizer']['state'].values()):
            raise ValueError('Every actual neural parameter must have complete optimizer steps')
        if api.torch.equal(s['model']['encoder.weight'], warm['model']['encoder.weight']):
            raise ValueError('Actual encoder change required')
        if not all(api.torch.isfinite(v).all() for v in s['model'].values()):
            raise ValueError('Finite trained parameters required')
        for k in ('input_hash', 'target_hash', 'proposal_hash', 'priority_mask_hash', 'warm_prediction_hash', 'trace'):
            api.base.exact(r[k], s[k])
        states[r['job'], r['arm']] = s; fits.append(r)
    for job in {r['job'] for r in fits}:
        api.assert_matched(*(states[job, arm] for arm in api.ARMS))
    ref = source.read(PUBLIC/'first_full_refit.json')
    api.parent.parent.parent.router.exact_state(states[fits[0]['job'], fits[0]['arm']],
        api.base.read_checkpoint(source.checked(ref['checkpoint'])))
    v = source.read(PUBLIC/'verification.json')
    if not v['all3_predictions_choices_metrics_exact'] or not v['all_parent_controls_exact'] or v['analysis_sha256'] != source.sha(PUBLIC/'analysis.json'):
        raise ValueError('Exact independent prediction/choice/readout verification required')
    source.once(PUBLIC/'fit_audit.json', dict(actual_neural_heads=6, primary_updates=12000,
        additional_full_refit_updates=2000, pilot_updates_including_repeats=400,
        actual_optimizer_and_common_encoder_checked=True, all_pairs_matched=True,
        first_full_refit_exact=True, verification=v, reporter=source.artifact(Path(__file__).resolve())))
    return states


def fitting_losses(states):
    _, _, data, jobs, refs, parts, points, routers, _, fitted, _, warm_refs = run.register(lambda **_: None)
    rows = []
    for job in jobs:
        d, warm = run.fitting(data, job, refs, parts, points, routers, fitted, warm_refs)
        c = d['causal']; choice = c['allowed'].argmax(1); ix = source.np.arange(len(choice))
        weights = run.parent.parent.api.query_weights(c, warm['sampler']['counts'])
        norm = api.parent.parent.normalizer(d['preprocess'])
        truth = (api.parent.parent.signed(d['truth_by_arm']['bound_aware']).numpy()/norm)[ix, choice]
        priority = d['priority_mask'].numpy()[ix, choice]
        for arm, state in [('warm', warm)]+[(a, states[job['key'], a]) for a in api.ARMS]:
            if arm != 'warm':
                for key in ('priority_mask_hash', 'warm_prediction_hash'):
                    api.base.exact(state[key], d[key])
            prediction = api.predict(state, c)[ix, choice]/c['scales'][:, None]/norm
            square = (prediction-truth)**2
            anchor = float((weights*square.mean(1)).sum())
            extra = float((weights*(square*priority).mean(1)).sum())
            movement = []
            for j, name in enumerate(('all', 'easy')):
                known = (weights > 0) & d['observed_future']
                current = known & (prediction[:, j] <= 0) & (truth[:, j] > 0)
                old = priority[:, j] & known
                movement.append(dict(channel=name, warm_false_safe=int(old.sum()),
                    warm_errors_remaining=int((old & current).sum()),
                    warm_errors_corrected=int((old & ~current).sum()),
                    new_false_safe=int((current & ~old).sum()), current_false_safe=int(current.sum()),
                    current_false_safe_query_weight=float(weights[current].sum()),
                    output_sign_diagnostic_not_complete_policy=True))
            rows.append(dict(job=job['key'], arm=arm, full_anchor_MSE=anchor,
                full_priority_MSE=extra, full_weight10_MSE=anchor+9*extra,
                known_rows=int(((weights > 0) & d['observed_future']).sum()),
                bound_rows=int(((weights > 0) & ~d['observed_future']).sum()),
                priority_channels=priority.sum(0).tolist(), empirical_query_weight_mass=float(weights.sum()),
                error_movement=movement,
                sign_decomposition=decompose(prediction, truth, weights, d['observed_future']),
                resubstitution_not_generalization=True))
        del d, warm; source.gc.collect()
    source.once(PUBLIC/'full_fitting_losses.json', source.stable(rows))
    return rows


def main():
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    states = audit(); losses = fitting_losses(states)
    a = source.read(PUBLIC/'analysis.json'); fitting = source.read(PUBLIC/'fitting_readout.json')
    held = [r for ref in a['group_receipts'] for r in source.read(source.checked(ref))['rows']]
    reliability = {scope: {arm: {subset: pool_reliability(rs, arm, subset) for subset in ('proposals', 'admitted')}
        for arm in ('kinematic', *api.ARMS)} for scope, rs in (('fit', fitting), ('held_SOURCE', held))}
    flow = {scope: {arm: pool_flow([dict(action_flow=r['action_flow'][arm]) for r in rs])
        for arm in api.ARMS} for scope, rs in (('fit', fitting), ('held_SOURCE', held))}
    source.once(PUBLIC/'mechanism_summary.json', source.stable(dict(reliability=reliability, flow=flow)))
    runtime = {p: source.read(PUBLIC/(p+'_execution.json')) for p in ('pilot', 'train', 'evaluate', 'verify')}
    source.once(PUBLIC/'runtime_summary.json', runtime)
    loss_table = ['| Context | Arm | Full anchor MSE | Fixed priority MSE | Combined weight10 MSE |', '|---|---|---:|---:|---:|']
    for r in losses:
        loss_table.append(f"| {r['job']} | {r['arm']} | {r['full_anchor_MSE']:.9f} | {r['full_priority_MSE']:.9f} | {r['full_weight10_MSE']:.9f} |")
    contrasts = ['| Scope | Contrast | Known pp | Completion lower pp | Completion upper pp |', '|---|---|---:|---:|---:|']
    for scope, records in a['contrasts'].items():
        for name, r in records.items():
            contrasts.append(f'| {scope} | {name} | '+' | '.join(number(r[k]['mean']) for k in
                ('known_pp', 'completion_lower_pp', 'completion_upper_pp'))+' |')
    s = a['held_policy']['false_safe_priority_veto']['safety']; train = runtime['train']
    text_once(PUBLIC/'README.md', f'''# Fixed Warm-Error Priority Training

I tested a single objective change: give the warm model's observed false-safe
fitting errors weight 10, retaining the full original MSE anchor. Both arms
start from identical kinematic neural weights and receive 2,000 updates. The
fixed error mask is supervision only, never an inference feature. Unknown
outcomes keep their original bounds and are not reclassified as observations.
This is standard hard-example replay, not a novel dynamics architecture.

## Execution

fresh_run: six neural risk fits, 12,000 primary updates, another 2,000-update
full refit, a 400-update resume pilot, inference and exact readout replay.
Training: {train['seconds']:.2f} seconds, peak RSS {train['peak_RSS_bytes']/1e9:.3f} GB,
native arm64 CPU, four compute threads, one interop thread, zero data workers.
All six fits freeze before evaluation.
cached_verified: source inputs, warm neural models, forecasts, router,
normalizers, old rules and all parent predictions/choices/metrics.
not_run: new trajectory dynamics, independent H, new producer seeds,
full expanded study or deployment. One-seed exposed SOURCE is development.
The initial zero-target synthetic gradient fixture failure is retained in
[the preflight amendment](preflight_amendment.md); it was fixed before training.

## Full Fitting Loss

Exact empirical-query weighted full-data losses, not minibatch scores.
The two objectives differ; compare both anchor and priority components, not
unlike total losses. Neither loss nor weighted score is a calibrated risk.

{chr(10).join(loss_table)}

## Held SOURCE Results

{table(a['held_policy'])}

{chr(10).join(contrasts)}

Prospective expansion condition: {a['expansion_supported']}.
Full original priority risk/support pass: {s['full_original_risk_pass']}.
Worst defined signed recording easy-net degradation: {number(s['worst_defined_signed_easy_net_percent'])}%.
No favorable context is chosen as a replacement result. See all fixed
contexts and admission diagnostics in [failure analysis](failure_analysis.md).

Seven already exposed SOURCE units, 17 recording views, one producer seed (17).
Completion bounds address missing labels, not sampling confidence intervals.
The original 24-group primary stays NA, not passed. Original 2% harm, easy-net and
support rules are unchanged. Empty matched controls remain unavailable;
no identity certificate. Deployment and submission readiness remain false.

[All metrics](analysis.json), [actual weight audit](fit_audit.json),
[protocol](protocol.md), [reproducibility](reproducibility.md).
Native obs8/pred12 image pixels, not SDD t50, verified metric/seconds,
human gold, true 3D or foundation evidence. Stage5C and SMC remain disabled.
''')
    lines = ['# Priority Learning: Fitting And Held Risk', '',
        'The fixed expansion condition is '+str(a['expansion_supported'])+'.', '',
        '## Fitting Policy, Resubstitution Only', '', table(a['fitting_policy']), '',
        '## All Fixed Contexts', '', '| Context | Priority-control completion lower pp | Control all/easy violations | Priority all/easy violations | Priority undefined easy | Expansion |',
        '|---|---:|---|---|---:|---|']
    for r in a['per_context']:
        c, t = r['control_safety'], r['priority_safety']
        lines.append(f"| {r['job']} | {number(r['completion_lower_pp'])} | {c['known_all_violations']}/{c['known_easy_violations']} | "
            f"{t['known_all_violations']}/{t['known_easy_violations']} | {t['undefined_selected_easy']} | {r['expansion_condition']} |")
    lines += ['', '## Conditional Risk', '',
        'Pooled known-label native-coordinate sums over repeated context views, not',
        'independent samples, official source-macro ratios or calibration certificates.',
        'Unknown labels retain bounds in the complete evaluation.', '',
        '| Scope | Arm | Set | Known / unknown | Predicted all/easy excess | Actual all/easy excess |',
        '|---|---|---|---:|---|---|']
    for scope, arms in reliability.items():
        for arm, subsets in arms.items():
            for subset, r in subsets.items():
                lines.append(f"| {scope} | {arm} | {subset} | {r['known_rows']}/{r['unknown_rows']} | "
                    f"{number(r['predicted_excess_sum'][0])}, {number(r['predicted_excess_sum'][1])} | "
                    f"{number(r['actual_excess_sum'][0])}, {number(r['actual_excess_sum'][1])} |")
    lines += ['', '## Limits', '',
        'Fixed warm-error emphasis may miss new errors introduced by training and',
        'can discard useful actions. Weight 10 is fixed, not tuned to this readout.',
        'The warm error mask is defined by fitting-data resubstitution, not by an',
        'independent error-discovery split or out-of-fold risk predictions.',
        'Lower fitting loss, fewer interventions or a favorable source cannot',
        'replace complete original risk/support. Weighted outputs are not unbiased',
        'expected harm. Old online-admission, gradient-parity, quantile and subset',
        'failures remain negative; this single contrast does not erase them.', '',
        'Model training cannot create missing full easy-reference support or restore',
        'independence to exposed SOURCE. Original primary NA and all old failures',
        'remain. No new H, deployment change, Stage5C, SMC or submission-ready claim.']
    text_once(PUBLIC/'failure_analysis.md', '\n'.join(lines)+'\n')
    text_once(PUBLIC/'gates.md', f'''# Execution Versus Scientific Evidence

- Six real trained neural heads and 12,000 primary updates: complete.
- Real resume, first full refit and all saved inference/readouts: exact.
- Source isolation and fitting-only error mask: retained.
- Original 2% risk/support pass: {s['full_original_risk_pass']}.
- Prospective expansion: {a['expansion_supported']}.
- Original 24-group primary: NA, not passed.
- New independent H/multi-seed confirmation: not_run.
- Deployment/submission readiness/identity certification/Stage5C/SMC: false.
''')
    text_once(PUBLIC/'reproducibility.md', '''# Reproduction

Use native arm64 .venv-pytorch/bin/python. All phases use a process lock,
CPU4/interop1/workers0, checkpoints/heartbeat, a 5 GiB disk reserve and a
512 MiB private artifact cap.

```sh
.venv-pytorch/bin/python scripts/diagnose_m3w_moving_admission_loss.py
.venv-pytorch/bin/python scripts/run_m3w_moving_admission_priority.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_admission_priority.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_admission_priority.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_admission_priority.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_admission_priority.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_admission_priority.py
```

Completed phases are immutable. Do not rerun them to overwrite results.
Interrupted fitting uses the same phase with --resume. Private prerequisites
must match recorded hashes. Reproduce in a separate checkout/artifact namespace,
not by deleting the original valid checkpoints. Existing verified reports can
be rechecked with the unchanged reporter; it rejects output drift.
The first registration is preserved; registration_v2 is operative after the
documented synthetic-fixture amendment before real training.
''')


if __name__ == '__main__':
    main()
