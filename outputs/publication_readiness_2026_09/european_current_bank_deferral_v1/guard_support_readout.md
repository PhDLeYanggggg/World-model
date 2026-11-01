# Frozen Guard: Two-Sided Failure Diagnosis

**fresh_run diagnostic; public-scalar report reconstruction.** All 72 original
groups and 444 repeated recording/context/seed views are retained. Actual run:
PID 58218, 102.577865 s, peak 11,423,350,784 B.
The diagnostic was registered after the primary readout and before these counts.
It is label-aware SOURCE failure analysis, not a new causal policy or a primary
test. Future labels are used only to identify errors; they never define eligibility.

## Observed Errors

|Quantity|Repeated action occurrences|
|---|---:|
|Legal known easy actions|1,283,400|
|True signed-risk-safe known easy actions|808,680|
|True safe easy actions rejected by the fixed guard|664,755|
|Rejected safe easy actions with strictly positive true benefit|620,297|
|Known signed-risk-unsafe actions admitted, all known rows|75,663|

These repeat actions across candidates, contexts and cost seeds, not independent
examples. The last row is not an easy-only false-positive rate and is not the
number actually chosen by deferral. Safety here means only the observed two
signed-risk inequalities, not certified full-outcome or physical safety. The guard
rejects 82.20% of the available true safe easy action
occurrences. Rejection reasons overlap;their counts must not be added.

## Missing Reference Support

12/444 views, covering
4 unique source/record pairs, have positive
full known easy reference but no known easy action surviving the pointwise guard.
Their original legal banks retain 296
true signed-risk-safe easy occurrences, including
208 with positive benefit; all are rejected.
This is not evidence that the whole candidate bank has no useful forecast.

For 6 repeated views of locality082/record51,
all three known easy rows belong to singleton ORIGINAL TARGET queries. The
native query allocator cannot pool those anchors with a contemporaneous second
target to recover known easy support. A wider real-world neighborhood is not
being asserted empty. For the remaining blocked views, multi-target pooling may
change support; no joint impossibility is inferred from pointwise rejection alone.
Unknown easy outcomes cannot supply a known denominator for the unchanged
conservative certificate. This does not prove their realized reference is zero.

## Consequence For The Next Repair

Better published-loss ranking alone cannot undo an unchanged eligibility mask.
Uniformly tightening thresholds can further remove safe denominator anchors;
uniform relaxation can admit additional harm. The critic therefore needs a
two-sided repair: reduce false low-risk admissions AND recover accurately modeled
beneficial reference support. Any such fit must be separately registered, retain
the original 2% contract and be tested against the complete causal policy, not an
oracle-filtered subset. A label-aware feasible witness is a diagnosis, not a model
result. Partial-label uncertainty and observed cost errors remain separate.

The current comparison remains 5/9 fitting gates, not independently confirmed or
deployable. No new training, threshold, calibration, data role or deployment change
occurred in this diagnostic. No SDD t+50, metric, seconds-level, true 3D, foundation,
human-gold or new dynamics claim follows. Stage5C/SMC remain off.
