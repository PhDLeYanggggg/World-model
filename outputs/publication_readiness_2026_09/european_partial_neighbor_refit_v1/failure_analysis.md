# Failure Analysis

## Main Finding

The matched context-support repair does not produce a reliable incremental
forecast gain. Equal-locality ADE gain against the legacy neural control is
+0.012%, with exploratory95% interval[-0.374%,+0.412%]. Hard gain is-0.152%
[-0.628%,+0.324%], and all-query FDE gain is-0.181%[-0.604%,+0.258%]. The
preregistered primary and nonnegative-hard screens fail. This is not proof of
exact equivalence, but it does not support the claimed incremental benefit.

## What Improved, and What Did Not

Positive-easy ADE gains0.335% over the old neural model. That secondary result
must not hide12.888% easy degradation against CV, whose corresponding gain
interval is[-25.844%,-3.181%]. The repaired model has2.927% all-query ADE gain
over the training-selected causal baseline[0.708%,5.355%], but that is not
the incremental effect of this repair. Its all-query gain against CV is2.719%
with an interval crossing zero[-8.201%,9.822%]. No deployment promotion follows.

The training reference is damped_velocity_090 in folds0/1 and
history_ols4_velocity in fold2. They were selected on fitting localities, not
by looking at held-locality outcomes. An all-query chosen reference is not
necessarily the best floor for easy cases; the CV comparison remains visible.

## Matched Execution Rules Out Some Explanations

One fresh legacy4000-update control matches every original parameter and the
sampler state exactly after pilot resume. All nine new models have the same
initial parameter shapes and values, training draw counts, final sampler state,
loss weighting, schedule and baseline choice as their matched controls. No
held-locality training draws occurred. All legacy cached forecasts reproduce
exactly by fresh inference before the prediction freeze. This argues against
a runtime, cache alignment or sampling-budget mismatch as the explanation.

Seed-level ADE gains are-0.122%,+0.137%,+0.021%; all three locality intervals
cross zero. Fold1 is negative for all three seeds, while fold2 is positive
for all three. Training-context sensitivity remains. Seven locality point
gains are positive and five negative; the worst is eu-locality-082 at-1.539%.
These are dependent development views, not12 independent confirmatory trials.

## Two Concrete Information Limits

1. **Neighbor coverage trades off against history support.** Mean current
   neighbors increases6.762 to7.304, but mean valid history slots decreases
   54.099 to50.513. Of282,529 changed queries,215,971 lose valid slots,65,280
   gain slots and1,278 retain the count. Nearer partial tracks can evict
   complete histories under the fixed eight-neighbor budget. This is measured
   input behavior, not a proven causal explanation of the forecast result.

2. **Flattened tokens discard supplied track association.** Exchanging
   positions between two neighbors at selected past times leaves each time's
   point set unchanged. On256 source prefixes, all constructed track paths
   change while the trained model's maximum output coordinate change is only
   4.58e-5. No future labels are used. The model can still infer aggregate
   flow from geometry, but cannot directly use the supplied identity linkage.
   This is a demonstrated representation limitation, not demonstrated
   accuracy benefit from repairing it.

The changed-input slice has ADE gain-0.050%[-0.466%,+0.377%]. The unchanged-
input slice has-0.382%[-1.030%,+0.188%]; unchanged-input hard gain is-0.703%
[-1.443%,-0.072%]. Shared weights change even when a particular input does not.
Slice ratios are not additive components of the equal-locality primary.
No subgroup is selected as a replacement endpoint.

## Remaining Uncertainty

Detector tracking/localization noise, sparse useful interaction events,
finite training, the ego-motion output bound and source-domain shifts remain
possible explanations. This experiment does not isolate those causes. It
also changes membership, attention masks and conditioning together, so their
individual effects are not identified. More current agents cannot be assumed
to mean more useful predictive information. Native easy/hard cuts do not
represent a common physically calibrated difficulty across localities.

## Decision

Keep the old model and policies untouched; do not promote the partial-context
variant or reopen reserved outcomes. Do not continue a mask/threshold sweep.
The next bounded causal hypothesis is track-preserving temporal encoding
followed by interaction. A separate same-parameter candidate passes structural
tests but has not been trained. Test it against the frozen flat partial model
on the same geometry, seeds, sampler and loss before changing neighbor admission,
loss or smoothing. Its value must be established by fresh matched training,
not by this diagnosis. Stage5C and SMC remain disabled.
