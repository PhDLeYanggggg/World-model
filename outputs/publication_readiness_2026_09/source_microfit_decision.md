# Training-Only Microfit: Output Scale Versus Fitability

## Material Passport

Mode: authorized code experiment. Role: training-only engineering diagnostic.
Inputs: verified source cache from the completed cost-dynamics comparison.
Status at registration: no training or results. No new held-site evaluation.

## Why This Check

All60 source trajectory heads fail to beat stationary CV on their complete
training complements, not only on held sites. Logged gradients are almost always
clipped, and matched-batch loss shows small persistent harm after zero-output
initialization. These observations motivate an optimization check, but do not
prove that decoder scale or clipping causes failure.

Hypothesis: multiplying every raw output by the observed context radius makes
the local output sensitivity poorly matched to the forecast-error scale. A
training-cost-scaled parameterization may fit supervised trajectories more
effectively while preserving the same observed context bound and ADE metric.

## Fixed Design

Use only the admitted source complement of bookstore, the first previously
registered source fold. Main training and sealed roles remain closed. Select
16 nonzero-label and16 zero-label examples with distinct scoped training-agent
IDs, a fixed hash-independent RNG order (seed104729), observed spatial support
and all training targets within0.8 times the observed context radius. This
future-label-based selection is legal only as training diagnostic construction;
it is never inference input, an official cohort filter or a benchmark result.

Two diagnostic cohorts:16 nonzero examples, and the same16 plus16 zero examples.
Each uses full-batch fitting, the same480-feature/past-RGB model, ADE objective,
AdamWlr0.0003/wd0.0001,gradientclip5 and2000updates. Seeds17/29/43 and two decoder
parameterizations give12fresh fits/24000updates. Final-budget checkpoints only.
The between-cohort contrast changes composition and batch size; it is not an
isolated estimate of a zero-target prevalence effect. Within-cohort decoder
comparisons are matched in every other respect.

For raw local network vector z, observed radius R and observed rotation Q:

- Original: Q R z/(1+norm(z)).
- Conditioned: Q c z/(1+(c/R)norm(z)), with c the full training-complement mean
  CV ADE, fixed before fitting and identical in every diagnostic arm.

Both are zero-initialized and have the same context-ball bound. Their origin
output Jacobians scale with R and c respectively. Evaluation remains in the
unchanged parent coordinate and normalized ADE. c is a training-fitted global
parameter, never a per-row future feature. No target endpoint, scene label,
test statistic or held-source label enters the model or scale fitting.

## Interpretation And Execution

Report initial/final full-microcohort ADE, nonzero-target ADE, zero-target absolute
harm, per-seed traces, all-update gradient-clipping fraction and compute cost.
Predictions and checkpoints must replay exactly; completed resume is immutable.
A100update pilot is resumed inside the fixed budget. Native arm64CPU4/workers0;
expected runtime is minutes locally, with checkpoint/heartbeat. CREATE's saved
authentication/project-path blocker remains unresolved; remote jobs are unknown.
Do not start duplicate local or remote writers.

Fitting these few training rows is not a generalization, modality, downstream,
joint-intervention or deployment result. A failed microfit also does not prove
that the task is intrinsically unlearnable. Keep the completed60-model negative
experiment intact. No main metric/split change, held-score tuning, Stage5C orSMC.
