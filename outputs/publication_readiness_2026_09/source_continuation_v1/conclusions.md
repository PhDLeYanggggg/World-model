# Learning-Rate Decay Repairs a Small Part of Training Fit, Not Generalization

## Material Passport

`fresh_run`: six real Torch continuation branches,48,000 new updates and24 fixed
full-training evaluations. Three verified parent models contain6,000 unique
inherited updates; they are not six independent fresh initializations.
`cached_verified`: source inputs, parent weights and split/schema identities.
Held-source/main forecasting and sealed evaluation: `not_run` by design.

The full trainer log spans39.5696minutes. Summed continuation work is2380.2700
seconds including fixed training evaluations and the100-update timing pilot.
All15,430 complete stationary-history rows in the source complement ofbookstore
remain:545 scoped agent IDs,29 recordings and4 source sites. This is not all SDD
training or the primary main benchmark. No previously scored file is overwritten.

## Results

All numbers below concern the same data used to fit the models. Positive means
lower training ADE than stationary CV; seed ranges are not generalization CIs.

| Fixed endpoint | Mean training gain | Three-seed range | Moving-target gain | Hard-target gain | Zero-target absolute harm, annotation pixels |
| --- | ---: | --- | ---: | ---: | ---: |
| Inherited step2000 | -1.5700% | [-2.0541%, -1.1895%] | -0.3069% | -0.0352% |0.025656 |
| Constant LR, step10000 | -1.0310% | [-1.2130%, -0.8424%] | +1.2446% | +0.8747% |0.046223 |
| Cosine LR, step10000 | +0.2533% | [+0.2039%, +0.3259%] | +0.8762% | +0.4155% |0.012653 |

Longer constant-rate training is not enough: all three branches remain worse
than CV overall. Decay yields a small positive training gain in all three seeds.
It reduces zero-target harm substantially compared with the constant schedule,
but also sacrifices some moving-target benefit. Thus the aggregate improvement
is not evidence that decay learns stronger movement dynamics than the constant
schedule. Both extended schedules improve moving-target fitting relative to the
inherited step2000 state, so this is also not merely a claim of exact zero-output
collapse. Every new update clips gradients in all six branches; clipping alone
does not prevent this small gain.

In parent-normalized units, constant-rate zero-target harm contributes+25.66085
to mean excess loss, while moving-target benefit contributes-14.03475. Under
decay these contributions are+7.02456 and-9.88071. Their sums exactly reconstruct
the aggregate loss differences. The primary distinction is gain/harm allocation,
not a large new predictive effect. Native improvement under decay is only about
0.002856 annotation pixels in mean ADE, and annotation noise is not calibrated.

The future-informed training-only binary oracle rises from0.0816% inherited
headroom to1.6945% under constant continuation and0.9988% under decay. These are
diagnostic choices between CV and the fitted forecast using training labels,
not a learned policy, an inference feature or held-scene headroom.

## What the Evidence Supports

- The previous full-training failure is partly schedule-dependent. It cannot be
  attributed only to a disconnected training path or universally useless inputs.
- At this budget, constant-rate exposure and cosine continuation differ with
  matched parents, sampling, loss, model and data. This isolates a schedule
  intervention, not its unique mechanism: effective AdamW shrinkage changes too.
- Mean cumulative exposure rises from8.2955 to41.4776 draws per training row.
  That is still not the2000 passes of the small memorization diagnostic. Neither
  convergence nor generalization follows from reaching10000 optimizer steps.
- A small positive aggregate training number does not protect simple cases.
  Zero-target harm remains positive; its percentage degradation is undefined
  because the CV denominator is zero. No easy-preservation pass is claimed.

Exactly55.52% of complete training targets stay entirely zero, and every future
waypoint separately has a zero majority. The triangle inequality establishes
that zero is optimal among fixed input-independent paths in stored normalized
coordinates. This is not an impossibility theorem for conditional predictors,
nor for a constant local code rotated/rescaled by an observed frame. It helps
explain why blindly predicting small movement can hurt, not why visual information
must be absent. See the exact coefficients in[analysis](analysis.json).

## Next Discriminating Experiment

Freeze all six final predictors rather than promote the best seed. In a separate
preregistered source diagnostic, test whether this optimization improvement
survives the excluded source site, retaining both schedules and all seeds. Add a
matched-budget coverage-only control before attributing benefit to RGB. These
source sites have already been explored in the project, so such a comparison
would still be exploratory source validation, not untouched confirmation.

If gain does not transfer, diagnose fitting versus conditional information and
observation support; do not spend another large blind architecture search. If
it does, build strictly out-of-fold gain/harm targets for the joint-intervention
comparison. Main selection, independent scene calibration and confirmation stay
closed until their own requirements are met. This follow-up is not run here.

## Verification and Remaining Gates

All24 saved training forecasts replay exactly. All three schedule pairs share
parent states and cumulative sample streams. Completed resume adds zero updates
and preserves88 artifacts including parents, plus the report. Fifty-four focused
tests pass; the full legacy suite was not rerun. The original scientific plot
was visually checked. Training, replay and analysis processes all ended exit0.

Report SHA256:
`c5397708e74a88153bbfc04dc29334f283f8fdb1f5f31a927598c585d652acd1`.
Registration was committed at`db31a217` before training. Previous60-model
negative results remain unchanged. No predictor or deployment threshold is
promoted. Useful held-scene dynamics, independent risk calibration, joint
intervention benefit and a submission-ready contribution remain unproved.

Current verdict: `small_training_fit_gain_generalization_unproved`.
The goal stays active and unmet. No metric/seconds/true3D/foundation claim.
Offline supplied/generated annotations are not strict sensor-as-of or human
intention gold. No Stage5C execution orSMC.
