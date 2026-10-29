# TRAIN-Internal Shared Risk Learning

## Material Passport and Hypothesis

This is a fresh paired training experiment using cached_verified source TRAIN
packets. It is not a new official evaluation split. Outer development and the
independent selection/calibration/confirmation roles remain unopened by this run.
Prior full-TRAIN census shows two scaled known easy-risk violations on TRAIN
versus58/72 on exposed development;16/24 source/context groups have only two TRAIN
recordings. Hypothesis: shared cost learning across the four source TRAIN packets
with the same frozen forecasting/controller context improves recording transfer.
It may fail because coordinate normalization, source mismatch, unknown outcomes
or missing causal information remain. No success is assumed.

## Fixed Scope

Use all six context groups, four source localities per context, seeds17/29/43.
Each original TRAIN packet is partitioned by a fixed source+recording hash:
hold out ceil(recordings/4), at least one; fit on all others. No outcomes choose
the split. Refit every normalizer, support limit, loss scale and initialization
prior on inner-fit rows only. Holdout rows never enter gradient, preprocessing,
early stopping or model selection. Evaluate all fixed finals once after all108
fits are frozen. The parent source labels/easy definitions and forecast models
remain unchanged; past-only380 features do not include learned prior selector
scores. TRAIN packets are the only data read.

The four packets are pooled only inside this TRAIN-internal experiment. This
does not promote previously exposed source-locality results to independent
confirmation, replace outer split definitions or authorize their use in a new
official claim. All parent forecasting/controller exposure remains recorded.

## Three Arms

- Local:72 freshly fitted source-specific budget-scaled32-wide cost heads,
  using the existing training implementation on inner-fit rows.
- Shared global:18 shared32-wide heads, one per context/seed; common parameters,
  equal-source/query loss, one pair of global risk duals.
- Shared source:18 identical shared heads, but one risk-dual pair per source,
  preventing one source's average easy-risk slack from canceling another's.

Normalize inputs and five moment labels by inner-fit source statistics before
sharing; invert each source's cost scale for evaluation. A shared head has one
TRAIN-derived decoder prior and RMS scale. This is a pooling/partial-sharing
repair, not a claim that only one scalar changed relative to local heads. The
shared-global/source contrast isolates the dual grouping. Same architecture,
initialization and sampled queries are required between the two shared arms.

All models receive2000 updates. Each source contributes16 queries/update in all
arms:four local models together and each shared model see the same per-source
sampling histories and total row budget. Shared models perform64-query updates,
not four times the per-source exposure. Four local parameter sets versus one
shared set are reported. Unknown target rows are never fabricated or trained as
safe. All remain in inference and completion bounds.

No new JEPA, trajectory producer, residual, latent rollout or stochastic proposal
is trained. Outputs are expected gain/harm/reference moments, not new dynamics.
Positive selected harm/reference and easy preservation retain the2% requirement.

## Evaluation and Decision

Full fixed policies and same-count utility comparisons against local heads;
also shared-source versus shared-global. All72 source/seed views remain visible.
Compute paired completion-lower utility, signed score MSE, known and upper risks,
unknown support, switch rate, per-recording results and predicted/observed harm.
Average repeated views inside source locality before3000 paired locality draws.
These are TRAIN-internal nominal diagnostics, not independent significance or a
new population-risk certificate. Three head seeds are not three forecast seeds.

Same-count empty risk is not estimable, never safety success. Full candidate risk
and support are separate, following the approved prospective contract. A repair
is eligible for later development design only if both full/matched utility
intervals favor it versus local, full-policy safety/support passes every required
view, and signed-score MSE is not demonstrably worse. Shared-source additionally
needs nonnegative evidence versus shared-global. No favorable scene/seed removal,
threshold tuning, retrospective primary substitution or automatic deployment.

## Execution and Verification

Native arm64 CPU4, interop1, workers0. Stream verified packets from CREATE;
no remote training submission. Preserve10GiB free space, checkpoint cap256MiB.
Real pilot100 steps tests exact interrupted/direct resume and matched per-source
draws before full execution. Per-fit atomic checkpoints every100 steps; heartbeat
and process lock; explicit resume, no reset after transient transport failure.
12-hour process boundary allows continuation from checkpoints, not a smaller task.
Precommit code/config/split roster before fitting; freeze all finals before held
inference. Raw data, caches, weights and images stay outside Git.

Dataset-local, observation8/prediction12, raw-frame stride12; no metric/seconds,
human-gold, physical-safety, true3D, foundation or submission-ready claim. Legacy
scores remain exploratory. Stage5C and SMC remain disabled.
