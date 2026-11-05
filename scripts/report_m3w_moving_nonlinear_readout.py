"""Preserve all matched nonlinear-readout results and original scientific gates."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_nonlinear_readout as run
from scripts.report_m3w_moving_direct_tail_risk import number, pool_reliability
from scripts.report_m3w_moving_readout_refit import pool_flow
from scripts.report_m3w_moving_harm_refinement import text_once

source, PUBLIC = run.source, run.PUBLIC


def audit():
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_fits'] != 6 or freeze['primary_scalar_boosting_rounds'] != 1536 or not freeze['all_before_validation']:
        raise ValueError('All fixed real-data fits must precede evaluation')
    fits, states = [], {}
    for ref in freeze['fits']:
        fit = source.read(source.checked(ref)); state = run.api.joblib.load(source.checked(fit['checkpoint']))
        frozen = run.api.base.read_checkpoint(source.checked(fit['frozen_checkpoint']))
        run.api.base.exact(state['frozen'], frozen)
        if (state['identity']['registration_sha256'] != source.sha(run.REG)
                or state['rounds'] != 128 or state['new_encoder_updates'] != 0
                or state['new_adam_updates'] != 0 or state['risk_budget'] != .02
                or state['sklearn'] != run.api.sklearn.__version__):
            raise ValueError('Registered unchanged risk model required')
        for model in state['models']:
            if model.n_iter_ != 128 or model.do_early_stopping_ or len(model._predictors) != 128:
                raise ValueError('All actual scalar boosting rounds required')
            if any(not source.np.isfinite(p[0].nodes['value']).all() for p in model._predictors):
                raise ValueError('Finite learned tree values required')
        if len(state['models']) != 2 or state['stats']['latent_score_dimensions'] != 66:
            raise ValueError('Two scalar outputs and real 64-dimensional parent required')
        run.api.base.exact(fit['trace'], state['trace'])
        states[fit['job'], fit['arm']] = state; fits.append(fit)
    for job in {f['job'] for f in fits}:
        a, b = [states[job, arm] for arm in run.api.ARMS]
        for key in ('matrix_hashes', 'frozen', 'settings', 'fit_ids', 'source_sites', 'risk_budget', 'identity'):
            run.api.base.exact(a[key], b[key])
    repeat = source.read(PUBLIC/'first_full_refit.json')
    run.api.exact_fit(states[fits[0]['job'], fits[0]['arm']], run.api.joblib.load(source.checked(repeat['checkpoint'])))
    replay = source.read(PUBLIC/'verification.json')
    if (not replay['all3_inference_choice_and_metric_blocks_exact']
            or not replay['all_parent_controls_exact'] or replay['analysis_sha256'] != source.sha(PUBLIC/'analysis.json')):
        raise ValueError('Exact original inference/decision/evaluation replay required')
    source.once(PUBLIC/'fit_audit.json', dict(all6_actual_fits=True, actual_scalar_boosting_rounds=1536,
        all3_matched_feature_ablation_pairs=True, first_full_refit_exact=True, replay=replay,
        frozen_encoders_unchanged=True, new_encoder_updates=0, reporting_script=source.artifact(Path(__file__))))
    return fits


def table(policies):
    lines = ['| Policy | Known gain vs floor % | Hard gain % | Switch % | Full risk/support | All violations | Easy violations | Undefined selected easy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('conditioned_veto', 'mean_veto', 'quantile90_veto', 'readout_veto', 'score_only_veto', 'latent_score_veto'):
        p, s = policies[name], policies[name]['safety']
        lines.append(f"| {name} | {number(p['known']['mean'])} | {number(p['hard']['mean'])} | "
            f"{number(p['switch_percent']['mean'])} | {s['supported']}/{s['required']} | "
            f"{s['known_all_violations']} | {s['known_easy_violations']} | {s['undefined_selected_easy']} |")
    return '\n'.join(lines)


def main():
    fits = audit(); a = source.read(PUBLIC/'analysis.json')
    held = [r for ref in a['group_receipts'] for r in source.read(source.checked(ref))['rows']]
    fitting = source.read(PUBLIC/'fitting_readout.json')
    reliability = {scope: {arm: {subset: pool_reliability(rows, arm, subset)
        for subset in ('proposals', 'admitted')} for arm in ('readout', *run.api.ARMS)}
        for scope, rows in (('fit', fitting), ('held_SOURCE', held))}
    flow = {scope: {arm: pool_flow([dict(action_flow=r['action_flow'][arm]) for r in rows])
        for arm in run.api.ARMS} for scope, rows in (('fit', fitting), ('held_SOURCE', held))}
    source.once(PUBLIC/'mechanism_summary.json', source.stable(dict(reliability=reliability, flow=flow)))
    runtime = {phase: source.read(PUBLIC/(phase+'_execution.json')) for phase in ('pilot', 'train', 'evaluate', 'verify')}
    source.once(PUBLIC/'runtime_summary.json', runtime)
    train = runtime['train']; s = a['held_policy']['latent_score_veto']['safety']
    decision = ('The prespecified compute-allocation condition passes; it is not a deployment or scientific certification.'
        if a['expansion_supported'] else
        'The prespecified compute-allocation condition fails. Do not promote this variant or select a favorable context.')
    losses = ['| Context | Arm | Rows / unknown bounds | Initial MSE | Final MSE | Fitting reduction % |',
        '|---|---|---:|---:|---:|---:|']
    for f in fits:
        t = f['trace'][-1]
        losses.append(f"| {f['job']} | {f['arm']} | {f['stats']['rows']}/{f['stats']['unknown_bound_rows']} | "
            f"{t['before_fitting_MSE']:.9f} | {t['fitting_MSE']:.9f} | {number(t['fitting_loss_reduction_percent'])} |")
    paired = ['| Scope | Contrast | Known pp | Completion lower pp | Completion upper pp |', '|---|---|---:|---:|---:|']
    for scope, contrasts in a['contrasts'].items():
        for name, c in contrasts.items():
            paired.append(f'| {scope} | {name} | '+' | '.join(number(c[k]['mean'])
                for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp'))+' |')
    text_once(PUBLIC/'README.md', f'''# Nonlinear Risk Readout: Matched Representation Control

I tested whether frozen neural features contain risk information that the
previous affine output missed. Both arms use standard histogram gradient
boosting: score_only sees two signed risk scores; latent_score sees those
scores plus the same 64 hidden features. The intervention changes the risk
model, not the trajectory, policy thresholds or evaluation contract.

## Evidence Scope

fresh_run: six two-output readout fits, 1,536 scalar boosting rounds; real
resume pilot, one complete independent refit, inference/readout and replay.
cached_verified: neural encoders, SVD readouts, proposal router, source roles,
causal arrays, original forecasts and all parent policies.
not_run: new encoder/trajectory training, independent H, new seeds, the full
108-direction study or deployment. This is not new Torch world-model training.

Training phase: {train['seconds']:.2f} seconds, {train['peak_RSS_bytes']/1e9:.3f} GB
peak RSS. Native arm64, single process; numerical thread pools are limited to
four. All six models freeze before held-source evaluation.
The first evaluation attempt failed on a nested-preprocessor metadata lookup.
The [explicit interface amendment](evaluation_interface_repair.md) preserves
that failure and all fitted weights, and restores the unchanged exclusion check.
It does not revise a result, source role or scientific criterion.

## Fit And Held-Source Results

{chr(10).join(losses)}

{table(a['held_policy'])}

{chr(10).join(paired)}

These are source-macro descriptive effects on {a['sources']} exposed SOURCE
units, {a['source_views']} source views and 17 recording views with one producer
seed. Completion lower/upper bounds are not confidence intervals. No CI,
independent significance or multi-seed stability is claimed for this pilot.

{decision}
Full original latent-policy risk/support: {s['full_original_risk_pass']}.
Worst defined signed recording easy-net degradation:
{number(s['worst_defined_signed_easy_net_percent'])}%, not a mean risk ratio.
Missing nonempty matched source views:
score_only={a['unavailable_matched_source_views']['score_only']},
latent_score={a['unavailable_matched_source_views']['latent_score']}.
The old 24-group primary remains NA, not passed. No identity-fallback
certification, retrospective success or removed failed view.

[Failure analysis](failure_analysis.md), [full numeric results](analysis.json),
[fixed protocol](protocol.md), [fit audit](fit_audit.json), [reproduction](reproducibility.md).
Deployment and submission readiness remain unchanged/false. Native obs8/pred12
image pixels, not SDD t50, verified seconds, metric, human-gold, true 3D or
foundation evidence. Stage5C and SMC remain disabled.
''')
    lines = ['# Conditional Readout Failure And Mechanism Analysis', '', decision, '',
        '## What This Comparison Can Identify', '',
        'Both arms retain the exact neural representation, offset, supervision,',
        'query weights, leaf/round budget, unknown-label bounds and decision rules.',
        'Only access to 64 hidden columns differs. This is a standard nonlinear',
        'readout feature ablation, not a comparison of newly trained encoders.',
        'A gain over the affine control alone does not show latent contribution:',
        'the score-only nonlinear control is required. Neither gives a transferred',
        'multicalibration or safety theorem. Tree complexity and effective fit',
        'capacity can still differ when feature dimension changes.', '',
        '## Result-Specific Diagnosis', '',
        'The hidden-feature arm improves fitting MSE in all three contexts and',
        'has a positive fitting utility contrast against the score-only arm.',
        'That advantage does not transfer: its held known utility contrast is',
        f"{number(a['contrasts']['held']['latent_minus_score']['known_pp']['mean'])} pp, with completion bounds "
        f"[{number(a['contrasts']['held']['latent_minus_score']['completion_lower_pp']['mean'])}, "
        f"{number(a['contrasts']['held']['latent_minus_score']['completion_upper_pp']['mean'])}] pp.",
        'Those bounds address missing outcomes, not sampling uncertainty. This',
        'is no demonstrated downstream benefit of the frozen latent, not proof',
        'that all neural representations or the underlying data are inadequate.', '',
        'Both readouts intervene much less than the affine control, losing net',
        'beneficial actions as well as harmful ones. Every registered context',
        'has a negative latent-minus-affine completion lower contrast. The',
        'latent arm retains 13 all-risk and 15 easy-risk violations across 17',
        'recording views; two selected-easy views additionally become undefined.',
        'A lower worst signed easy-net degradation does not pass the separate',
        'positive-harm budget or manufacture reference support.', '',
        'The action-flow sums below distinguish this from simple over-rejection:',
        'the retained and newly admitted actions still have positive observed',
        'all/easy budget excess despite negative predicted excess. The model',
        'has not learned the admission boundary reliably. Rejected net benefit',
        'cannot justify reopening those actions without a new valid risk model.', '',
        '## Actual Original Risk Checks', '', table(a['fit_policy']), '',
        'The preceding table is fitting-only, not validation. The held table is',
        'in the main report. Lower fitting MSE cannot replace either gate.', '',
        '## Conditional Risk Error', '',
        'Pooled available-label sums in native image units, repeated across context',
        'views. They are not independent observations or certification. Unknown',
        'labels retain their bounds in the full policy evaluation, not here.', '',
        '| Scope | Model | Set | Known | Unknown | Predicted all/easy excess | Actual all/easy excess |',
        '|---|---|---|---:|---:|---|---|']
    fmt = lambda values: ', '.join(number(v) for v in values)
    for scope, models in reliability.items():
        for arm, subsets in models.items():
            for subset, r in subsets.items():
                lines.append(f"| {scope} | {arm} | {subset} | {r['known_rows']} | {r['unknown_rows']} | "
                    f"{fmt(r['predicted_excess_sum'])} | {fmt(r['actual_excess_sum'])} |")
    lines += ['', '## Which Proposals Changed?', '',
        '| Scope | Model | Flow vs affine | Known | Unknown | Net benefit | Positive harm | All excess | Easy excess |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for scope, models in flow.items():
        for arm, parts in models.items():
            for name, f in parts.items():
                lines.append(f"| {scope} | {arm} | {name} | {f['known_rows']} | {f['unknown_rows']} | "
                    f"{number(f['known_net_benefit'])} | {number(f['known_moments'][1])} | "
                    f"{number(f['known_all_budget_excess'])} | {number(f['known_easy_budget_excess'])} |")
    lines += ['', '## Every Registered Context', '',
        '| Context | Latent minus affine completion lower pp | Affine all/easy violations | Latent all/easy violations | Expansion condition |',
        '|---|---:|---|---|---|']
    for c in a['per_context']:
        before, after = c['original_safety'], c['latent_safety']
        lines.append(f"| {c['job']} | {number(c['direct_lower_pp'])} | {before['known_all_violations']}/{before['known_easy_violations']} | "
            f"{after['known_all_violations']}/{after['known_easy_violations']} | {c['expansion_condition']} |")
    lines += ['', '## Limits', '',
        'New fitting error reduction is not independent generalization or a new',
        'dynamics contribution. Inspect retained harmful actions and discarded',
        'beneficial actions together; fewer interventions alone are not success.',
        'The same source is used for the fixed encoder and its readout fitting,',
        'so fitting diagnostics are resubstitution, not calibration evidence.',
        'Held SOURCE is already exposed development, not fresh confirmation.', '',
        'Any absence of easy-reference mass remains NA under the original rule.',
        'Changing a nonlinear score cannot mathematically create missing labels',
        'or reference support. No favorable source, seed or checkpoint is selected',
        'after this readout. Any next model needs a new frozen comparison.', '',
        'No metric, seconds, human-gold, physical-safety, true3D, foundation or',
        'submission-ready claim. No new H; Stage5C and SMC remain off.']
    text_once(PUBLIC/'failure_analysis.md', '\n'.join(lines)+'\n')
    text_once(PUBLIC/'gates.md', f'''# Original Rules And Separate Evidence Gates

- Six real two-output risk readouts, 1,536 scalar rounds: complete.
- Real interrupted/uninterrupted pilot: both arms exact.
- First full fit independently reproduced: exact.
- Three complete prediction/action/readout blocks and old controls: exact.
- Frozen neural encoders and original data/input/target rules: unchanged.
- Full original latent risk/support pass: {s['full_original_risk_pass']}.
- Prespecified expansion condition: {a['expansion_supported']}.
- Original 24-group primary: NA, not passed.
- Independent confirmation / multi-seed result / new neural dynamics: not_run.
- Identity certification / deployment upgrade / submission ready: false.
- Stage5C / SMC: false.
''')
    text_once(PUBLIC/'reproducibility.md', '''# Reproduction And Recovery

Native arm64 `.venv-pytorch/bin/python`; Torch CPU4/interop1, workers0 and four
numerical threads during sklearn fit/inference. No resource probing or new HPC
job. Atomic checkpoints every eight boosting rounds. Use `--resume` only for an
interrupted incomplete phase. Completed receipts and old results are immutable.

First execution in the registered namespace:

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_nonlinear_risk_readout.py tests/test_m3w_nonlinear_readout_exclusion.py tests/test_m3w_nonlinear_readout_reporting.py tests/test_m3w_frozen_readout_refit.py tests/test_m3w_readout_refit_reporting.py
.venv-pytorch/bin/python scripts/run_m3w_moving_nonlinear_readout.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_nonlinear_readout.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_nonlinear_readout.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_nonlinear_readout.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_nonlinear_readout.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_nonlinear_readout.py
```

After completion, rerun the report command to check all weights, parent hashes,
model pairing and frozen replay receipt without overwriting results. A fresh
full independent execution needs a new registered output namespace. Private
SOURCE arrays, parent weights and banks are not included in public Git. Public
light evidence alone does not reproduce raw inference. No raw data, cache,
images, weights or third-party datasets are committed. Focused tests do not
stand in for the complete historical test suite.
''')


if __name__ == '__main__':
    main()
