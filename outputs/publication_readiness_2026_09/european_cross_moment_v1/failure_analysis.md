# Failure Analysis

## Supported Findings

1. Target weighting matters for some candidate/event combinations. Damping/all
   improves at both matched-count anchors in all nine conditional comparisons.
   Pair availability is identical, so this change is not explained by more pairs.
2. Weighting is not a universal neural repair. Batch neural/all is mixed;
   batch neural/easy has no positive full-policy interval. All neural all-ADE
   and hard-ADE points still lose to equally protected damping in both modes.
3. Fixed normalization has partial, heterogeneous effects. It improves several
   easy-event comparisons but retains negative intervals and small neural gains.
   It does not prove that random batch denominators caused the overall failure.
4. Lower fitting loss is insufficient. Total loss decreases in 30/36 and 33/36
   heads, without a stable neural-over-damping benefit.
5. Positive-easy preservation and zero-reference preservation are different.
   Neural worst positive-easy degradation stays below 0.18%, but each mode has
   twelve views with added error on observed zero-CV rows.

## Evidence Limits

Cross weights can be highly concentrated: up to 87.83% of a neural/easy audited
batch's pair weight lies on one pair. Fixed normalization removes one random
denominator, not dependence, heavy weights, target noise or representation shift.
Logged gradient summaries describe sampled reporting steps, not all updates.
No post-outcome weight clipping or tuning was performed.

Only four observed zero-reference cases exist, in one locality. Each has two
future labels, not a full twelve-step trajectory, and no endpoint. This was
freshly checked against the bound source arrays. It limits generalization of
both apparent failures and apparent protection; it does not justify erasing
the rows or changing the primary rule after seeing results.

The direct neural/damping comparison is of complete protected pipelines.
It still combines forecast quality and candidate-specific learnability. It
does not prove a universal inability of neural forecasting, identify a unique
causal mechanism, or demonstrate a general safety theorem for cross moments.

## Next Controlled Diagnostic

Use the already frozen predictions and policies to separate (a) available neural
advantage over protected damping, (b) missed beneficial interventions, and
(c) harmful accepted interventions. Stratify by locality and annotation support.
Any oracle remains an offline diagnostic. Retain the existing primary cohort
and add completeness sensitivity without changing inference inputs or declaring
the sensitivity the new winner. If reachable neural benefit is small, another
controller loss is unlikely to be the shortest path; change the trajectory
candidate in a newly registered experiment. If reachable benefit is substantial,
test a source-fitted support/uncertainty repair with the same risk constraint.

All data here are opened development. This is not independent confirmation,
historical Stage37 recertification or deployment evidence. Detector-track pixels
and raw-stride 8/12 only; no metric/seconds, human-gold, physical-safety, true-3D
or foundation claim. Reserved roles closed; Stage5C and SMC off.
