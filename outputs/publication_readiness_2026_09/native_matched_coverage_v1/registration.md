# Matched-Coverage Native Cost Diagnostic

## Material Passport

- Date: 2026-09-21. Fixed analysis before new matched-policy outcome calculation.
- Scope: post-hoc mechanism diagnosis on already explored source sites.
- Parent: native_gain_harm_v1, completed and independently verified.
- No fresh training, model selection, threshold sweep, calibration or deployment.

## Question

Does penalizing harm underestimation improve ranking of interventions, or does
its observed harm reduction mainly follow from switching less often? The parent
readout cannot isolate these explanations because the two neural losses switch
different numbers of rows. This is not independent statistical confirmation.

## Fixed Comparison

Reuse every frozen source prediction and cost score, four sites, three seeds,
all 175,756 query indices. Keep eight observed/twelve predicted annotation steps,
native pixel ADE/FDE and equal mean of within-scene CV-relative gains.

For each site/seed, take the intervention count K from each of the two already
fixed underharm4 rules: positive predicted gain; positive gain plus predicted
harm <= 0.1 benefit. K depends only on predicted scores and causal equality to
CV, never future labels or their support. No new K or threshold is searched.

At exactly that K, among all rows whose candidate differs from CV, compare:

1. Original underharm4 rule (anchor).
2. Neural MSE ratio ranking: (benefit - harm)/(benefit + harm).
3. Neural MSE predicted net gain ranking.
4. Underharm4 predicted net gain ranking.
5. Signed ridge predicted net gain ranking; preserve negative harm estimates.
6. One fixed query-ID shuffle, independent of scores/outcomes (hash control).

Top K ties use the global query ID, not outcomes. For zero total predicted cost,
the ratio is defined as zero; this score convention does not create a percentage
performance gain. Exact underharm4 ratio top-K must reproduce the anchor as a
sanity check. Outcomes, future masks and future easy/hard groups are never inputs
to any selection function. Unknown-label rows remain in the choice population.

Also report the analytic expectation under uniform random selection of K eligible
rows. This expected-error control is not a deployed fractional trajectory and
has no claimed realized-policy tail quantiles or safe-seed certificate.

## Analysis

Primary paired contrast: anchor minus MSE ratio **gain percentage points**, using
the same per-scene CV denominators and paired scene bootstrap, three seeds kept.
Report both coverage budgets; do not select a favorable budget or seed.
Secondary fixed rankers remain visible regardless of direction. Also report
ADE/FDE, hard q75/positive-easy q25 diagnostics, strict complete zero-CV absolute
harm, known selected harm, intervention count, unknown selected outcomes, tails
and worst scene. Cut values are from the same bound training-only preprocessors.

Resample the four physical scenes 3,000 times after averaging errors over seeds.
Do not count overlapping windows as independent samples. The four sites remain
design-exposed; intervals are conditional developmental summaries, not a new
test or a calibrated risk guarantee. Scene-wide ranking uses a full batch of
past-input queries; it is a diagnostic matched-budget allocation, not an online
deployment rule. Outcomes do not determine the batch budget or ordering.

Verify that anchor metrics reproduce the parent, all policies have identical
per-view K, all source bindings remain fixed, and choices are invariant to future
label/support changes. Preserve the preexisting strict zero-reference criterion;
do not relax it if all nontrivial controls fail.

## Execution and Boundaries

Local native arm64 CPU4/workers0 is sufficient for cached arithmetic. Run the
existing hash/schema audit before any new matched readout. Save config, code
hashes, selected-ID hashes, summaries and resume/verifier receipt. No new GPU
allocation, CREATE job or raw-data download is required for this diagnostic.
Original val/test, main/external and bookstore roles stay closed. No future
endpoint, central velocity or test goal enters input. No metric, seconds-level,
foundation or true-3D claim. Stage5C and SMC remain disabled.
