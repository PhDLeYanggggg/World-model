"""Audit real paired neural fits and report original risk outcomes unchanged."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_kinematic_risk as run
from scripts.report_m3w_moving_direct_tail_risk import number, pool_reliability
from scripts.report_m3w_moving_readout_refit import pool_flow
from scripts.report_m3w_moving_harm_refinement import text_once

source, PUBLIC = run.source, run.PUBLIC
NAMES = ('conditioned_veto', 'mean_veto', 'quantile90_veto', 'readout_veto', 'masked_veto', 'kinematic_veto')


def table(policies):
    out = ['| Policy | Known floor gain % | Hard gain % | Switch % | Full risk/support | All/easy violations | Undefined selected easy |',
        '|---|---:|---:|---:|---:|---|---:|']
    for name in NAMES:
        r = policies[name]; s = r['safety']
        out.append(f"| {name} | {number(r['known']['mean'])} | {number(r['hard']['mean'])} | "
            f"{number(r['switch_percent']['mean'])} | {s['supported']}/{s['required']} | "
            f"{s['known_all_violations']}/{s['known_easy_violations']} | {s['undefined_selected_easy']} |")
    return '\n'.join(out)


def audit():
    freeze = source.read(PUBLIC/'training_freeze.json'); states, fits = {}, []
    if freeze['complete_heads'] != 6 or freeze['primary_updates'] != 12000 or not freeze['all_before_validation']:
        raise ValueError('All six actual neural fits required')
    for ref in freeze['fits']:
        fit = source.read(source.checked(ref)); state = run.api.base.read_checkpoint(source.checked(fit['checkpoint']))
        warm = run.api.base.read_checkpoint(source.checked(fit['warm_checkpoint']))
        if state['step'] != 2000 or state['parameters'] != 32130 or state['risk_budget'] != .02:
            raise ValueError('Actual fixed update budget and unchanged risk rule required')
        run.api.base.exact(state['warm_model'], warm['model'])
        for key in ('model', 'initial_model'):
            if not all(run.api.torch.isfinite(v).all() for v in state[key].values()):
                raise ValueError('Finite real neural weights required')
        if any(int(v['step']) != 2000 for v in state['optimizer']['state'].values()):
            raise ValueError('Every parameter must have actual optimizer state')
        if run.api.torch.equal(state['model']['encoder.weight'], warm['model']['encoder.weight']):
            raise ValueError('Neural encoder updates must be real')
        if bool(state['model']['kinematics.weight'].any()) != (state['arm'] == 'kinematic'):
            raise ValueError('Only unmasked extra fields may train the new branch')
        for key in ('trace', 'parameters', 'extra_hash', 'extra_raw_hash', 'common_input_hash', 'target_hash', 'proposal_hash'):
            run.api.base.exact(fit[key], state[key])
        states[fit['job'], fit['arm']] = state; fits.append(fit)
    for job in {f['job'] for f in fits}:
        run.api.assert_matched(*(states[job, a] for a in run.api.ARMS))
    repeat = source.read(PUBLIC/'first_full_refit.json')
    run.api.parent.parent.router.exact_state(states[fits[0]['job'], fits[0]['arm']],
        run.api.base.read_checkpoint(source.checked(repeat['checkpoint'])))
    verification = source.read(PUBLIC/'verification.json')
    if not verification['all3_predictions_choices_metrics_exact'] or not verification['all_parent_controls_exact'] or verification['analysis_sha256'] != source.sha(PUBLIC/'analysis.json'):
        raise ValueError('Exact independent saved inference and readout replay required')
    source.once(PUBLIC/'fit_audit.json', dict(actual_neural_heads=6, primary_encoder_updates=12000,
        additional_full_refit_updates=2000, pilot_updates_including_repeats=400,
        original_sampler_targets_and_warm_predictions_matched=True, first_full_refit_exact=True,
        verification=verification, reporting_script=source.artifact(Path(__file__).resolve())))
    return fits, states


def full_fitting_loss(states):
    _, _, data, jobs, refs, parts, points, routers, _, fitted, _ = run.register(lambda **_: None)
    results = []
    for job in jobs:
        d, warm = run.fitting(data, job, refs, parts, points, routers, fitted)
        c = d['causal']; choice = c['allowed'].argmax(1); rows = source.np.arange(len(choice))
        weights = run.parent.api.query_weights(c, warm['sampler']['counts'])
        normalizer = run.api.parent.normalizer(d['preprocess'])
        truth = (run.api.parent.signed(d['truth_by_arm']['bound_aware'])/run.api.torch.from_numpy(normalizer)).numpy()[rows, choice]
        def loss(pred):
            normed = pred[rows, choice]/c['scales'][:, None]/normalizer
            return float((weights*((normed-truth)**2).mean(1)).sum())
        initial = loss(run.api.parent.predict(warm, c))
        for arm in run.api.ARMS:
            state = states[job['key'], arm]
            for key in ('extra_preprocess', 'extra_raw_hash', 'extra_hash'):
                run.api.base.exact(state[key], d[key])
            actual = loss(run.api.predict(state, c))
            results.append(dict(job=job['key'], arm=arm, warm_fitting_MSE=initial, final_fitting_MSE=actual,
                fitting_reduction_percent=100*(1-actual/initial) if initial else None,
                known_proposal_rows=int(((weights > 0) & d['observed_future']).sum()),
                bound_proposal_rows=int(((weights > 0) & ~d['observed_future']).sum()),
                empirical_query_weight_mass=float(weights.sum()), resubstitution_not_generalization=True))
        del d, warm; source.gc.collect()
    source.once(PUBLIC/'full_fitting_loss.json', source.stable(results))
    return results


def main():
    run.api.torch.set_num_threads(4); run.api.torch.set_num_interop_threads(1)
    fits, states = audit(); a = source.read(PUBLIC/'analysis.json')
    losses = full_fitting_loss(states)
    held = [r for ref in a['group_receipts'] for r in source.read(source.checked(ref))['rows']]
    fitting = source.read(PUBLIC/'fitting_readout.json')
    reliability = {scope: {arm: {subset: pool_reliability(rows, arm, subset)
        for subset in ('proposals', 'admitted')} for arm in ('mean', 'readout', *run.api.ARMS)}
        for scope, rows in (('fit', fitting), ('held_SOURCE', held))}
    flow = {scope: {arm: pool_flow([dict(action_flow=r['action_flow'][arm]) for r in rows])
        for arm in run.api.ARMS} for scope, rows in (('fit', fitting), ('held_SOURCE', held))}
    source.once(PUBLIC/'mechanism_summary.json', source.stable(dict(reliability=reliability, flow=flow)))
    runtime = {p: source.read(PUBLIC/(p+'_execution.json')) for p in ('pilot', 'train', 'evaluate', 'verify')}
    source.once(PUBLIC/'runtime_summary.json', runtime)
    loss_table = ['| Context | Arm | Warm MSE | Final MSE | Reduction % | Known / bound-only proposals |',
        '|---|---|---:|---:|---:|---:|']
    for r in losses:
        loss_table.append(f"| {r['job']} | {r['arm']} | {r['warm_fitting_MSE']:.9f} | {r['final_fitting_MSE']:.9f} | "
            f"{number(r['fitting_reduction_percent'])} | {r['known_proposal_rows']}/{r['bound_proposal_rows']} |")
    paired = ['| Scope | Contrast | Known pp | Completion lower pp | Completion upper pp |', '|---|---|---:|---:|---:|']
    for scope, contrasts in a['contrasts'].items():
        for name, c in contrasts.items():
            paired.append(f'| {scope} | {name} | '+' | '.join(number(c[k]['mean'])
                for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp'))+' |')
    train = runtime['train']; s = a['held_policy']['kinematic_veto']['safety']
    verdict = ('The prespecified expansion condition passes. It is not a scientific or deployment certificate.'
        if a['expansion_supported'] else 'The prespecified expansion condition fails. No favorable context is selected for promotion.')
    text_once(PUBLIC/'README.md', f'''# Causal Kinematic Risk Representation

I compared explicit backward motion, turn and neighbor-approach features with
an otherwise identical continued-training control. Both start from the same
neural weights; the new branch starts at zero, preserving initial predictions.
All common encoder weights learn. The masked arm receives no new fields.
The motion information already exists in the flat input; this tests its
representation, not a newly acquired modality or a new trajectory model.

## Execution And Evidence Scope

fresh_run: six real Torch heads, 12,000 primary updates, another 2,000-update
independent refit, 400-update real resume pilot, inference and exact replay.
Training took {train['seconds']:.2f} seconds with {train['peak_RSS_bytes']/1e9:.3f} GB
peak RSS, native arm64 CPU4/interop1/workers0. All six fits precede evaluation.
cached_verified: forecast bank, router, prior neural weights, original source
roles, preprocessing, targets and parent control predictions/metrics.
not_run: new trajectory training, new independent H, new producer seeds, full
108-direction expansion or deployment. One-seed exposed SOURCE is development.

## Full Fitting Loss

These are fresh full fitting-data losses, using the exact empirical query
weights and original bound-aware normalized targets. They are not single
minibatch scores, validation metrics or uncertainty calibration.

{chr(10).join(loss_table)}

## Held SOURCE Outcomes

{table(a['held_policy'])}

{chr(10).join(paired)}

{verdict}
Full original kinematic risk/support pass: {s['full_original_risk_pass']}.
Worst defined signed recording easy-net degradation:
{number(s['worst_defined_signed_easy_net_percent'])}%.

Results are source-macro descriptive effects on {a['sources']} already exposed
SOURCE units, {a['source_views']} source views, 17 recording views and seed17.
Completion bounds address missing labels, not sampling confidence intervals.
No CI or independent significance is claimed. The original 24-group primary
remains NA, not passed; empty controls remain unavailable and no identity-
fallback certificate is added. Original 2% harm, easy-net and support rules
are unchanged. Deployment and submission readiness are unchanged/false.

[Failure and mechanism analysis](failure_analysis.md), [input trace](input_diagnosis.md),
[protocol](protocol.md), [all metrics](analysis.json), [weight audit](fit_audit.json),
[reproduction](reproducibility.md).

Native obs8/pred12 image pixels, not SDD t50, verified metric/seconds,
human-gold, true3D or foundation evidence. Stage5C and SMC remain disabled.
''')
    lines = ['# Kinematic Representation: Mechanism And Limits', '', verdict, '',
        '## Result-Specific Diagnosis', '',
        'The kinematic arm does not beat its equal-budget masked control on',
        'source-macro known utility, nor the earlier affine readout. Only one',
        'of three fixed contexts meets the prospective expansion condition;',
        'the positive context is retained but not selected as a new winner.', '',
        'Known all-risk violations increase from 13 in the masked control to',
        '14 in the kinematic arm. Known easy violations fall from 16 to 14,',
        'but two selected-easy views also become undefined; this count change',
        'does not establish preserved easy risk. Full support/risk remains',
        '0/17, and worst defined signed easy-net degradation is still 37.35%.', '',
        'Extra neural training without the features also loses held utility',
        'against its original warm model. The failure therefore cannot be',
        'credited solely to the new branch. The controlled feature contrast',
        'is needed to separate continued training from representation effects.', '',
        '## Fitting Is Not Generalization', '', table(a['fitting_policy']), '',
        'The table above is resubstitution only. Better fitting loss or utility',
        'does not certify the risk of admitted actions. The held table and all',
        'contrasts are in the main report; no favorable source is removed.', '',
        '## All Registered Contexts', '',
        '| Context | Kinematic minus masked completion lower pp | Masked all/easy violations | Kinematic all/easy violations | Expansion |',
        '|---|---:|---|---|---|']
    for c in a['per_context']:
        b, n = c['masked_safety'], c['kinematic_safety']
        lines.append(f"| {c['job']} | {number(c['completion_lower_pp'])} | {b['known_all_violations']}/{b['known_easy_violations']} | "
            f"{n['known_all_violations']}/{n['known_easy_violations']} | {c['expansion_condition']} |")
    lines += ['', '## Conditional Risk', '',
        'Pooled known-label native-coordinate sums over repeated context views;',
        'not independent observations, official source-macro ratios or safety',
        'certification. Unknown outcomes retain bounds in the full evaluation.', '',
        '| Scope | Arm | Set | Known / unknown | Predicted all/easy excess | Actual all/easy excess |',
        '|---|---|---|---:|---|---|']
    fmt = lambda xs: ', '.join(number(x) for x in xs)
    for scope, arms in reliability.items():
        for arm, subsets in arms.items():
            for subset, r in subsets.items():
                lines.append(f"| {scope} | {arm} | {subset} | {r['known_rows']}/{r['unknown_rows']} | "
                    f"{fmt(r['predicted_excess_sum'])} | {fmt(r['actual_excess_sum'])} |")
    lines += ['', '## Action Flow Relative To Affine Control', '',
        '| Scope | Arm | Flow | Known / unknown | Net benefit | Positive harm | All/easy budget excess |',
        '|---|---|---|---:|---:|---:|---|']
    for scope, arms in flow.items():
        for arm, parts in arms.items():
            for name, r in parts.items():
                lines.append(f"| {scope} | {arm} | {name} | {r['known_rows']}/{r['unknown_rows']} | "
                    f"{number(r['known_net_benefit'])} | {number(r['known_moments'][1])} | "
                    f"{number(r['known_all_budget_excess'])}/{number(r['known_easy_budget_excess'])} |")
    lines += ['', '## What Is And Is Not Identified', '',
        'The matched treatment changes a deterministic causal feature basis.',
        'Both arms continue training from the same warm model, with matching',
        'source roles, optimizer reset, target, sampling and update budget.',
        'The masked branch has less effective capacity even with equal nominal',
        'parameter count. This does not isolate every architecture mechanism.', '',
        'Closest approach uses extrapolated observed motion; no measured TTC,',
        'collision or physical-validity guarantee is inferred. No new scene',
        'image, video representation or goal label is added. A failure does not',
        'prove that all temporal or interaction representations are useless.', '',
        'Model changes cannot create missing full easy-reference support.',
        'Undefined support remains NA under the binding unchanged contract.',
        'Repeated development comparisons cannot become independent evidence.',
        'No deployment upgrade, new H, Stage5C, SMC or submission-ready claim.']
    text_once(PUBLIC/'failure_analysis.md', '\n'.join(lines)+'\n')
    text_once(PUBLIC/'gates.md', f'''# Separate Execution And Scientific Outcomes

- Real neural encoder training: six heads, 12,000 primary updates complete.
- Initial identity, real resume, full first refit: exact.
- All three prediction/action/metric blocks and old controls: exact.
- Source isolation and fitting-only feature preprocessing: retained.
- Original 2% risk/support pass: {s['full_original_risk_pass']}.
- Prospective compute expansion: {a['expansion_supported']}.
- Original 24-group primary: NA, not passed.
- Independent confirmation and multi-seed study: not_run.
- Deployment upgrade / submission readiness: false.
- Identity certification / Stage5C / SMC: false.
''')
    text_once(PUBLIC/'reproducibility.md', '''# Reproduction

Use native arm64 .venv-pytorch, CPU4/interop1/workers0; no resource probing.
Registered definitions and receipts are immutable. The first execution is:

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_kinematic_risk.py tests/test_m3w_direct_tail_risk.py tests/test_m3w_kinematic_feature_provenance.py
.venv-pytorch/bin/python scripts/run_m3w_moving_kinematic_risk.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_kinematic_risk.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_kinematic_risk.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_kinematic_risk.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_kinematic_risk.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_kinematic_risk.py
```

Interrupted incomplete phases use --resume with the same registered settings.
Completed receipts are not overwritten. The report can rerun to audit actual
weights, lineage, full fitting losses and exact saved readout. A new independent
execution needs a separate registered namespace. Private SOURCE annotations,
forecast banks and checkpoints are not in public Git; light reports alone
cannot reproduce raw inference. Focused checks do not replace the full suite.
''')


if __name__ == '__main__':
    main()
