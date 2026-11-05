"""Audit and report the frozen-encoder control without changing its readout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_readout_refit as run
from scripts.report_m3w_moving_direct_tail_risk import number, pool_reliability
from scripts.report_m3w_moving_harm_refinement import text_once

source, PUBLIC = run.source, run.PUBLIC


def pool_flow(rows):
    result = {}
    for name in ('retained', 'newly_rejected', 'newly_admitted', 'still_rejected'):
        parts = [row['action_flow'][name] for row in rows]
        result[name] = dict(known_rows=sum(p['known_rows'] for p in parts),
            unknown_rows=sum(p['unknown_rows'] for p in parts),
            known_moments=source.np.sum([p['known_moments'] for p in parts], axis=0).tolist(),
            **{k: sum(p[k] for p in parts) for k in ('known_net_benefit',
                'known_all_budget_excess', 'known_easy_budget_excess')},
            repeated_rows_not_independent=True)
    return result


def audit():
    frozen = source.read(PUBLIC/'training_freeze.json')
    if (frozen['complete_readouts'] != 3 or not frozen['all_before_validation']
            or frozen['new_encoder_updates'] != 0 or frozen['new_adam_updates'] != 0):
        raise ValueError('Exactly three completed frozen-encoder fits required')
    fits, states = [], []
    for ref in frozen['fits']:
        fit = source.read(source.checked(ref))
        s = run.api.base.read_checkpoint(source.checked(fit['checkpoint']))
        warm = run.api.base.read_checkpoint(source.checked(fit['warm_checkpoint']))
        if (s['identity']['registration_sha256'] != source.sha(run.REG)
                or s['learned_parameters'] != 130 or s['new_encoder_updates'] or s['new_adam_updates']
                or s['optimizer_state'] is not None or s['risk_budget'] != .02):
            raise ValueError('Completed registered affine-only repair required')
        for key in ('fit_ids', 'source_sites', 'preprocess', 'settings', 'seed',
                'input_hash', 'target_hash', 'proposal_mask_hash'):
            run.api.base.exact(s[key], warm[key])
        run.api.base.exact(s['warm_model'], warm['model'])
        for key in warm['model']:
            if not key.startswith('excess.'):
                run.api.base.exact(s['model'][key], warm['model'][key])
        if not all(run.api.torch.isfinite(v).all() for v in s['model'].values()):
            raise ValueError('Non-finite refitted weights')
        d = s['diagnostics']
        if (d['retained_rank'] != 65 or d['relative_normal_residual'] > 1e-8
                or d['refit_actual_float32_MSE'] > d['original_actual_float32_MSE'] + 1e-5):
            raise ValueError('Observed solve must be full rank and reduce actual fitting loss')
        run.api.base.exact(fit['diagnostics'], d)
        fits.append(fit); states.append(s)
    repeat = source.read(PUBLIC/'first_exact_refit.json')
    run.api.exact_refit(states[0], run.api.base.read_checkpoint(source.checked(repeat['checkpoint'])))
    verification = source.read(PUBLIC/'verification.json')
    if (verification['analysis_sha256'] != source.sha(PUBLIC/'analysis.json')
            or not verification['all3_predictions_choices_and_readouts_exact']
            or not verification['all_parent_controls_exact']):
        raise ValueError('Exact fresh readout replay required')
    source.once(PUBLIC/'fit_audit.json', dict(complete_readout_fits=3,
        encoder_bitwise_unchanged=True, actual_fitting_MSE_reduced_all3=True,
        original_fitting_inputs_targets_proposals_exact=True, first_complete_refit_exact=True,
        new_encoder_updates=0, new_adam_updates=0, verification=verification,
        reporting_script=source.artifact(Path(__file__))))
    return fits


def policy_table(policies):
    lines = ['| Policy | Known floor gain % | Hard gain % | Switch % | Full support | All violations | Easy violations | Undefined selected easy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('public_router', 'conditioned_veto', 'mean_veto', 'quantile90_veto', 'readout_veto'):
        p, s = policies[name], policies[name]['safety']
        lines.append(f"| {name} | {number(p['known']['mean'])} | {number(p['hard']['mean'])} | "
            f"{number(p['switch_percent']['mean'])} | {s['supported']}/{s['required']} | "
            f"{s['known_all_violations']} | {s['known_easy_violations']} | {s['undefined_selected_easy']} |")
    return '\n'.join(lines)


def main():
    fits = audit(); a = source.read(PUBLIC/'analysis.json')
    held = [row for ref in a['group_receipts'] for row in source.read(source.checked(ref))['rows']]
    fitting = source.read(PUBLIC/'fitting_readout.json')
    reliability = {scope: {arm: {sub: pool_reliability(rows, arm, sub)
        for sub in ('proposals', 'admitted')} for arm in ('mean', 'readout')}
        for scope, rows in (('fit', fitting), ('held_SOURCE', held))}
    flows = {scope: pool_flow(rows) for scope, rows in (('fit', fitting), ('held_SOURCE', held))}
    source.once(PUBLIC/'mechanism_summary.json', source.stable(dict(reliability=reliability, flow=flows)))
    runtime = {phase: source.read(PUBLIC/(phase+'_execution.json')) for phase in ('train', 'evaluate', 'verify')}
    source.once(PUBLIC/'runtime_summary.json', runtime)
    train = runtime['train']; p = a['held_policy']['readout_veto']; s = p['safety']
    contrast = a['held_readout_minus_mean']
    decision = ('The prespecified allocation condition is met; a separate full experiment is still required.'
        if a['expansion_supported'] else
        'The prespecified allocation condition is not met. Do not promote this output-only repair.')
    losses = ['| Frozen context | Before MSE | After MSE | Reduction % | Rank | Relative normal residual |',
        '|---|---:|---:|---:|---:|---:|']
    for fit in fits:
        d = fit['diagnostics']
        losses.append(f"| {fit['job']} | {d['original_actual_float32_MSE']:.9f} | "
            f"{d['refit_actual_float32_MSE']:.9f} | {d['fit_loss_reduction_percent']:.6f} | "
            f"{d['retained_rank']}/65 | {d['relative_normal_residual']:.3e} |")
    paired = ['| Scope, readout minus mean | Known pp | Completion lower pp | Completion upper pp |',
        '|---|---:|---:|---:|']
    for name in ('fit', 'held'):
        c = a[name+'_readout_minus_mean']
        paired.append(f'| {name} | '+' | '.join(number(c[k]['mean']) for k in
            ('known_pp', 'completion_lower_pp', 'completion_upper_pp'))+' |')
    text_once(PUBLIC/'README.md', f'''# Frozen Neural Readout: Optimization Control

I kept the original evaluation rules and changed only the two affine risk
outputs of three frozen neural representations. Weighted float64 SVD refits
130 parameters per model using the original empirical query weights. Actual
saved float32 inference is checked separately. This is a standard optimization
control, not a new trajectory model, loss invention or safety certificate.

## Evidence Scope

- fresh_run: three real-data output solves, one independent complete repeat,
  causal inference, original policy readout and exact independent replay.
- cached_verified: original neural encoders, forecasts, source roles, causal
  features, source-specific preprocessing, proposal support and parent policies.
- not_run: new encoder training, full 108-context-direction comparison,
  additional seeds, independent H confirmation or deployment.

Training phase: {train['seconds']:.2f} seconds, peak RSS
{train['peak_RSS_bytes']/1e9:.3f} GB. There were zero new Adam updates.
All three output systems have retained rank 65; encoders remain bitwise fixed.
This pilot has {a['sources']} SOURCE units, {a['source_views']} source views,
17 recording views and one producer seed. These are already exposed development
sources, not independent confirmation. No CI or multi-seed effect is claimed.

## Fitting Loss And Original Policy Evaluation

{chr(10).join(losses)}

{policy_table(a['held_policy'])}

{chr(10).join(paired)}

{decision}
Full original risk/support pass: {s['full_original_risk_pass']}. Worst defined
signed recording easy-net degradation: {number(s['worst_defined_signed_easy_net_percent'])}%.
The nonempty matched comparison is unavailable in
{a['unavailable_matched_source_views']}/{a['source_views']} source views.
The original 24-group primary remains NA, not passed; these pilot tables do
not replace it. No identity-fallback certification is introduced.

See [failure analysis](failure_analysis.md), [complete results](analysis.json),
[fit audit](fit_audit.json), [gates](gates.md) and [reproduction](reproducibility.md).
Deployment and submission readiness remain false. Native obs8/pred12 image
pixels are not verified seconds, metric units, SDD t50, true 3D or foundation
evidence. Labels are not human gold. Stage5C and SMC remain disabled.
''')
    lines = ['# Optimization Versus Admission Error', '', decision, '',
        '## Controlled Intervention', '',
        'The encoder, proposals, causal features, normalization, fitting targets,',
        'empirical query weights and thresholds are fixed. Only the affine output',
        'is solved. This controls final-output optimization, not encoder capacity',
        'or the changing-encoder AdamW path. Decoupled weight decay is not part',
        'of this least-squares data objective. Lower fitting loss is a numerical',
        'result, not evidence of selected-set calibration or deployment safety.', '',
        '## Selected Risk Error', '',
        'Values below are pooled available-label sums in local image units across',
        'repeated contexts. All/easy are two distinct columns. Unknown-label bounds',
        'remain in fitting and original evaluation, but are never called observed',
        'outcomes. These sums are not independent test samples or risk ratios.', '',
        '| Scope | Head | Set | Known rows | Unknown rows | Predicted all/easy excess | Actual all/easy excess |',
        '|---|---|---|---:|---:|---|---|']
    fmt = lambda values: ', '.join(number(v) for v in values)
    for scope, arms in reliability.items():
        for arm, subsets in arms.items():
            for sub, r in subsets.items():
                lines.append(f"| {scope} | {arm} | {sub} | {r['known_rows']} | {r['unknown_rows']} | "
                    f"{fmt(r['predicted_excess_sum'])} | {fmt(r['actual_excess_sum'])} |")
    lines += ['', '## Which Actions Changed?', '',
        'The same frozen proposal is either admitted or rejected; trajectory and',
        'amplitude cannot change. Benefit is positive error reduction, harm is',
        'positive error increase. Neither cancels the other in the risk gate.', '',
        '| Scope | Flow | Known rows | Unknown rows | Known benefit | Known harm | Net benefit | All excess | Easy excess |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope, parts in flows.items():
        for name, f in parts.items():
            lines.append(f"| {scope} | {name} | {f['known_rows']} | {f['unknown_rows']} | "
                f"{number(f['known_moments'][0])} | {number(f['known_moments'][1])} | "
                f"{number(f['known_net_benefit'])} | {number(f['known_all_budget_excess'])} | "
                f"{number(f['known_easy_budget_excess'])} |")
    lines += ['', '## All Registered Contexts', '',
        '| Context | Completion lower vs mean pp | Original all/easy violations | Refit all/easy violations | Expand |',
        '|---|---:|---|---|---|']
    for c in a['per_context']:
        before, after = c['original_safety'], c['refit_safety']
        lines.append(f"| {c['job']} | {number(c['direct_lower_pp'])} | "
            f"{before['known_all_violations']}/{before['known_easy_violations']} | "
            f"{after['known_all_violations']}/{after['known_easy_violations']} | {c['expansion_condition']} |")
    bf, af = a['fit_policy']['mean_veto']['safety'], a['fit_policy']['readout_veto']['safety']
    lines += ['', '## What The Result Establishes', '',
        f"Held known floor gain moves from {number(a['held_policy']['mean_veto']['known']['mean'])}% "
        f"to {number(p['known']['mean'])}%, a {number(contrast['known_pp']['mean'])}pp descriptive change.",
        f"The direct completion-lower change is {number(contrast['completion_lower_pp']['mean'])}pp, "
        'but two of three registered context lower contrasts remain negative.',
        f"Held all/easy violations stay at {s['known_all_violations']}/{s['known_easy_violations']}; "
        f"full risk/support stays {s['supported']}/{s['required']}.",
        f"Fitting all/easy violations rise from {bf['known_all_violations']}/{bf['known_easy_violations']} "
        f"to {af['known_all_violations']}/{af['known_easy_violations']} across {af['required']} repeated views.",
        'Thus final-affine optimization can recover a small amount of mean utility',
        'in this pilot, but it does not fix admission risk. This does not establish',
        'statistical significance, disprove end-to-end representation learning,',
        'or prove that additional causal information cannot help.', '',
        '## Interpretation Limits And Next Decision', '',
        'A least-squares solution makes the weighted residual orthogonal to its',
        'linear design columns. The nonlinear admission indicator is not generally',
        'one of those columns. It therefore does not establish zero residual or',
        'safe positive harm on the model-selected subset, even on fitting data.',
        'That algebraic distinction is not a proof of an irreducible data limit.', '',
        'Compare fitting and held-source error before attributing failure only',
        'to domain shift. If output optimization leaves fitting admission error,',
        'the next repair needs decision-relevant conditional information or',
        'conditional-risk supervision, not merely more steps on this last layer.',
        'Any such repair needs its own prospective model specification and must',
        'be distinguished from the already-failed admission penalties and event',
        'heads. No retrospective threshold search is justified by this table.', '',
        'Unobserved future labels and absent reference mass remain separate limits.',
        'The original full-sample support obstruction is not fixed by a smaller',
        'regression error. No identity certification, revised threshold or deleted',
        'empty control is allowed. All old negative studies remain negative.', '',
        'No new H, independent significance, physical safety, metric, seconds,',
        'human-gold, true 3D or foundation claim. Stage5C and SMC remain off.']
    text_once(PUBLIC/'failure_analysis.md', '\n'.join(lines)+'\n')
    text_once(PUBLIC/'gates.md', f'''# Original Rules, Separate Engineering And Scientific Gates

- Three output fits and one exact repeat: complete.
- Encoder, source roles, input/target hashes and proposals unchanged: verified.
- All three inference, decision and original readout blocks replay: verified.
- Actual fitting loss reduced: yes; not a scientific gate.
- Full original risk/support: {s['full_original_risk_pass']}.
- Known held utility change vs mean: {number(contrast['known_pp']['mean'])} pp.
- Completion-lower change vs mean: {number(contrast['completion_lower_pp']['mean'])} pp.
- Prespecified pilot expansion: {a['expansion_supported']}.
- Original 24-group primary: NA, not passed.
- New independent confirmation and full three-seed result: not_run.
- Identity certification / deployment changed / submission ready: false.
- Stage5C / SMC: false.
''')
    text_once(PUBLIC/'reproducibility.md', '''# Reproduce And Verify

Use native arm64 `.venv-pytorch/bin/python`, CPU4/interop1/workers0. The runtime
records PID, heartbeat, peak memory and immutable phase receipts. All private
dependencies, checkpoints and cache hashes must match the registered parents.

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_frozen_readout_refit.py tests/test_m3w_readout_refit_reporting.py tests/test_m3w_direct_tail_risk.py tests/test_m3w_direct_tail_reporting.py
.venv-pytorch/bin/python scripts/run_m3w_moving_readout_refit.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_readout_refit.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_readout_refit.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_readout_refit.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_readout_refit.py
```

Completed experiment phases intentionally reject overwriting receipts. Use
`--resume` only on an interrupted incomplete phase; completed fits are reused
after binding checks. The report command re-audits checkpoints and hashes
without retraining or using a new holdout. A new independent raw-data rerun
requires a separately registered namespace, not deleting frozen artifacts.
Four primary solves include three fits and one reproduction, not four seeds.
Private licensed data and pretrained parents are required: public Git alone
does not reconstruct them. No raw data, feature/cache arrays, weights or third-
party material is committed. Focused tests are not the full historical suite.
''')


if __name__ == '__main__':
    main()
