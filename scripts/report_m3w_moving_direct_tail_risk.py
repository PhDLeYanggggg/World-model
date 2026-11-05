"""Report the frozen direct-tail pilot, preserving failed and undefined views."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_direct_tail_risk as run
from scripts.report_m3w_moving_harm_refinement import text_once

source, PUBLIC = run.source, run.PUBLIC


def number(value):
    return 'NA' if value is None else f'{value:.6f}'


def audit():
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_heads'] != 6 or freeze['primary_updates'] != 12000 or not freeze['all_before_validation']:
        raise ValueError('Complete registered pilot training required')
    fits = [source.read(source.checked(ref)) for ref in freeze['fits']]
    states, rows = {}, []
    for fit in fits:
        s = run.api.base.read_checkpoint(source.checked(fit['checkpoint']))
        c = run.api.base.read_checkpoint(source.checked(fit['cached_control']))
        if (s['identity']['registration_sha256'] != source.sha(run.REG)
                or s['step'] != 2000 or s['budget'] != 2000 or s['parameters'] != 28034):
            raise ValueError('Exact registered real checkpoint required')
        for key in ('preprocess', 'sampler', 'draw_hash', 'fit_ids', 'source_sites'):
            run.api.base.exact(s[key], c[key])
        a = run.api.base.read_checkpoint(source.checked(s['identity']['router_checkpoint']))
        if set(a['source_sites']) & set(s['source_sites']) or set(a['fit_ids']) & set(s['fit_ids']):
            raise ValueError('Training router may not overlap risk fitting sources')
        if not all(int(v['step']) == 2000 for v in s['optimizer']['state'].values()):
            raise ValueError('Actual Adam updates must equal 2000')
        if not all(run.api.torch.isfinite(v).all() for v in s['model'].values()):
            raise ValueError('Finite trained state required')
        states[fit['job'], fit['arm']] = s
        rows.append(dict(job=fit['job'], arm=fit['arm'], real_updates=2000,
            frozen_control_sampling_preprocessing_exact=True, source_excluded_proposer=True,
            checkpoint=fit['checkpoint']))
    for job in {f['job'] for f in fits}:
        run.api.assert_matched(states[job, 'mean'], states[job, 'quantile90'])
    repeat = source.read(PUBLIC/'first_full_refit.json')
    first = fits[0]
    run.api.parent.router.exact_state(states[first['job'], first['arm']],
        run.api.base.read_checkpoint(source.checked(repeat['checkpoint'])))
    verification = source.read(PUBLIC/'verification.json')
    if verification['analysis_sha256'] != source.sha(PUBLIC/'analysis.json'):
        raise ValueError('Frozen readout must have an exact independent replay')
    source.once(PUBLIC/'fit_audit.json', dict(rows=rows, all6_real_complete=True,
        all3_loss_pairs_matched=True, first_full_refit_exact=True, replay=verification,
        reporting_script=source.artifact(Path(__file__))))
    return fits


def pool_reliability(rows, arm, subset):
    parts = [r['reliability'][arm][subset] for r in rows]
    n = sum(p['known_rows'] for p in parts)
    return dict(known_rows=n, unknown_rows=sum(p['unknown_rows'] for p in parts),
        predicted_excess_sum=source.np.sum([p['predicted_excess_sum'] for p in parts], axis=0).tolist(),
        actual_excess_sum=source.np.sum([p['actual_excess_sum'] for p in parts], axis=0).tolist(),
        underprediction_fraction=(sum(source.np.asarray(p['underprediction_fraction'])*p['known_rows']
            for p in parts if p['known_rows'])/n).tolist() if n else None,
        repeated_rows_not_independent=True)


def main():
    fits = audit(); a = source.read(PUBLIC/'analysis.json')
    held = [source.read(source.checked(ref)) for ref in a['group_receipts']]
    fit = source.read(PUBLIC/'fitting_diagnostics.json')
    reliability = {scope: {arm: {sub: pool_reliability(rows, arm, sub)
        for sub in ('proposals', 'admitted')} for arm in run.api.ARMS}
        for scope, rows in (('fit', fit), ('held_SOURCE', held))}
    source.once(PUBLIC/'reliability_summary.json', source.stable(reliability))
    table = ['| Policy | Known gain vs floor (%) | Hard gain (%) | Switch (%) | Supported / required | All violations | Easy violations | Undefined selected easy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('public_router', 'all_action_veto', 'conditioned_veto', 'mean_veto', 'quantile90_veto'):
        p = a['policy'][name]; s = p['safety']
        table.append(f"| {name} | {number(p['known']['mean'])} | {number(p['hard']['mean'])} | {number(p['switch_percent']['mean'])} | "
            f"{s['supported']}/{s['required']} | {s['known_all_violations']} | {s['known_easy_violations']} | {s['undefined_selected_easy']} |")
    paired = ['| Paired strategy | Known (pp) | Completion lower (pp) | Completion upper (pp) |', '|---|---:|---:|---:|']
    for name, p in a['paired'].items():
        paired.append(f"| {name} | "+' | '.join(number(p[m]['mean']) for m in ('known_pp', 'completion_lower_pp', 'completion_upper_pp'))+' |')
    train = source.read(PUBLIC/'train_execution.json')
    runtime = {phase: source.read(PUBLIC/(phase+'_execution.json')) for phase in ('pilot', 'train', 'evaluate', 'verify')}
    source.once(PUBLIC/'runtime_summary.json', runtime)
    result = ('The prespecified pilot expansion conditions are met; this is permission to design a separate full study, not a scientific pass.'
        if a['pilot_expansion_supported'] else
        'The prespecified pilot expansion conditions fail. This loss variant is not promoted to a full study or deployment.')
    header = '\n'.join(table)+'\n\n'+'\n'.join(paired)
    text_once(PUBLIC/'README.md', f'''# Direct Tail Risk: Completed Model Repair Pilot

I kept the original evaluation rules and tested a model-side change: predict
signed all/easy harm-budget excess directly, with an unbounded linear output.
The two matched losses are mean squared error and fixed .9 quantile pinball.
Both models retain causal context, the source-excluded router and original 2%
budget. A predicted quantile is not a calibrated safety certificate.

## Evidence Scope

fresh_run: six real neural heads,12,000 primary optimizer updates;400 resume-
smoke updates and an independent2,000-update full refit are additional checks.
Training phase completed in{train['seconds']:.2f}s, peak RSS{train['peak_RSS_bytes']/1e9:.3f}GB.
cached_verified: original forecasts, own-source preprocessing and frozen policy
controls. All three new inference/readout blocks and cached controls replay.
not_run: full108-context-direction comparison, additional seeds, independent H,
new trajectory training, or deployment. The three fixed contexts all use seed17;
they are not three independent training seeds. There are{a['sources']}SOURCE
units and{a['source_views']}source views. The table is source-macro descriptive,
not an independent-test confidence interval. No favorable view is removed.

## Results

{header}

{result}
Nonempty matched primary unavailable: mean{a['matched_primary_unavailable']['mean']},
quantile{a['matched_primary_unavailable']['quantile90']}source views. The old24group
primary remainsNA, not passed. Full-strategy contrasts in the table do not
replace that primary. Positive-harm/support and signed easy-net remain distinct.

See[protocol](protocol.md),[all machine-readable results](analysis.json),
[failure analysis](failure_analysis.md),[reproducibility](reproducibility.md),
and[fit audit](fit_audit.json). Deployment and submission readiness remain false.
obs8/pred12 native annotation order, local-image pixels, non-human-gold labels;
not metric, verified seconds, SDDt50, true3D or foundation. Stage5C/SMC off.
''')
    lines = ['# Failure And Mechanism Analysis', '', result, '',
        '## What Was Tested', '',
        'Earlier signed-risk penalties on constrained cost moments, log-harm losses,',
        'event heads, and proposal-only supervision were not silently repeated or',
        'reclassified as successes. This is a narrower unbounded signed-output test.',
        'Mean versus quantile isolates the loss within that new architecture; either',
        'versus the old critic changes both parameterization and supervision.', '',
        '## Fitting Versus Held-Source Tail', '',
        'The following pooled known-row excess sums are descriptive, in local pixel',
        'units and repeated context views. Columns are all/easy; unknown bounds stay',
        'in the policy evaluation but are not misrepresented as realized outcomes.', '',
        '| Scope | Head | Set | Known rows | Predicted excess sum | Actual excess sum | Fraction actual > predicted |',
        '|---|---|---|---:|---|---|---|']
    for scope, arms in reliability.items():
        for arm, subsets in arms.items():
            for sub, r in subsets.items():
                fmt = lambda values: 'NA' if values is None else ', '.join(number(x) for x in values)
                lines.append(f"| {scope} | {arm} | {sub} | {r['known_rows']} | {fmt(r['predicted_excess_sum'])} | "
                    f"{fmt(r['actual_excess_sum'])} | {fmt(r['underprediction_fraction'])} |")
    lines += ['', '## Reading The Pilot', '',
        f"Mean minus old conditioned known utility: {number(a['paired']['mean_minus_conditioned']['known_pp']['mean'])}pp; "
        f"quantile minus old: {number(a['paired']['quantile_minus_conditioned']['known_pp']['mean'])}pp.",
        f"The quantile switch rate is {number(a['policy']['quantile90_veto']['switch_percent']['mean'])}%, "
        f"versus {number(a['policy']['conditioned_veto']['switch_percent']['mean'])}% for the old conditioned veto.",
        'These are the identical seven pilot SOURCE units, not the previous full',
        '23-source experiment. Neither point estimates nor this one seed establish',
        'independent significance. See every source/context contrast in analysis.', '',
        'The fit-versus-held table matters: if the admitted fitting subset already',
        'has underestimated excess, the failure cannot be attributed solely to',
        'unseen-scene shift. Marginal tail coverage and conditional admitted-set',
        'error are different. Underprediction of a negative budget excess does',
        'not itself mean harm; actual positive excess and full record gates must',
        'still be inspected. Conversely, low marginal underprediction frequency',
        'does not certify selected mean harm. See the pre-readout method note.', '',
        '## Limits And Next Decision', '',
        'Training fit, held prediction error, and policy safety are separate tests.',
        'A lower fraction of underpredictions is not by itself useful intervention:',
        'excessive abstention can lose benefit while still leaving selected harm or',
        'missing-support failures. Quantile .9 does not bound a selected-set mean.',
        'The observed signed excess also depends on rare easy-reference support.',
        'Sparse support is retained as NA, not turned into a certificate.', '',
        'This pilot is limited to the preselected contexts and one seed. A negative',
        'pilot does not disprove every tail learner or demonstrate irreducible',
        'future ambiguity. It does rule out promoting this particular repair from',
        'its fitting loss alone. Further work must distinguish lost beneficial',
        'actions from retained harmful actions using fixed causal slices before',
        'another model/threshold sweep. More computation is not evidence by itself.', '',
        'Original2%harm, support, easy-net and empty-control rules are unchanged.',
        'No new H, independent confirmation, physical-safety, metric, seconds,',
        'true3D or foundation claim. Stage5C/SMC remain off.']
    text_once(PUBLIC/'failure_analysis.md', '\n'.join(lines)+'\n')
    text_once(PUBLIC/'gates.md', f'''# Pilot Gates, Not Deployment Certification

- Real six-head training and matched objective-only comparison: pass.
- Exact smoke resume, first full refit and three readout replays: pass.
- Original source isolation and rules retained: pass.
- Full original mean-policy risk/support: {a['policy']['mean_veto']['safety']['full_original_risk_pass']}.
- Full original quantile-policy risk/support: {a['policy']['quantile90_veto']['safety']['full_original_risk_pass']}.
- Prespecified pilot expansion: {a['pilot_expansion_supported']}.
- Original24group primary: NA, not passed.
- Independent confirmation: not_run.
- Deployment changed / submission ready / Stage5C / SMC: false.

Engineering completion does not substitute for a scientific gate.
''')
    text_once(PUBLIC/'reproducibility.md', '''# Reproduce The Direct Tail Pilot

Use native arm64 `.venv-pytorch/bin/python`, CPU4/interop1/workers0.
The registration binds model, runner, config, protocol and tests before fitting.
The two losses use identical initialization, preprocessing, sampling, proposal
support and update count. No future outcome is supplied to inference.

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_direct_tail_risk.py tests/test_m3w_proposal_conditioned_risk.py tests/test_m3w_proposal_risk_runner.py
.venv-pytorch/bin/python scripts/run_m3w_moving_direct_tail_risk.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_direct_tail_risk.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_direct_tail_risk.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_direct_tail_risk.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_direct_tail_risk.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_direct_tail_risk.py
```

Use `--resume` only for an interrupted, incomplete phase. Completed phase
receipts are immutable and should not be removed to rerun them. Existing
checkpoints can be rechecked by the report audit; a fresh independent rerun
needs a new explicit output namespace, not overwriting these results.
The private SOURCE assets, forecast bank and pretrained parent heads are local
dependencies, not bundled with the public repository. Git alone is insufficient
to reproduce those data. Checkpoints and caches are intentionally not committed.
The complete historical test suite was not rerun for this scoped model change.
''')


if __name__ == '__main__':
    main()
