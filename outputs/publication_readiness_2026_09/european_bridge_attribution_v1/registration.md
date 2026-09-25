# Policy-Bridge Attribution: Fixed Development Experiment

## Material Passport
Fresh motion-only controller training and fixed-family readout; full-pair heads,
forecasts and data are cached_verified. No new backbone, calibration or test role.
This record is committed before fitting and before new policy decisions/readout.

## Questions
Does the aligned neural/ranked cost system beat a simple ridge system on exactly
the same candidate forecasts? Does it rank interventions better at the same
per-query count? Does access to neural trajectory candidates improve the final
forecast relative to a retrained motion-only controller?

The preceding all-risk bridge improved development ADE by 3.95%-5.57% over the
older conservative controller, with net easy degradation below 2%. Most pooled
gain was from motion-to-motion changes, and realized positive-harm caps failed.
These findings motivate attribution, not a claim of neural dynamics success.

## Fixed Pairs and Training
Full pair: reuse the exact easy old_stop R and all old_stop P from producer A.
Reuse its controller B utility/all-risk neural and ridge heads with hashes and
prefix replay. No full-pair refit or new easy-risk head.

Motion-only pair: R and P are A's easy and all protected motion floors, each
choosing CV or damping-0.97. Remove neural trajectories AND both neural-policy
bits from the feature schema. Motion-floor scorers remain learned neural models;
motion-only means no neural trajectory candidate, not no neural computation.
Fit new utility and all-risk heads on B only with the same 383-column schema,
2,000-update budget, seeds, row support, source-balanced sampling and CV cost
scale. Features and learned normalization legitimately differ between pairs.
Fit corresponding ridge moment regressors at alpha 0.01. There are 36 new Torch
heads (72,000 updates), 36 new ridge fits, and 36+36 cached full-pair fits.

Use all six ordered producer/controller assignments and seeds 17,29,43. No
selection/readout fitting. A's entire forecast/floor producer chain excludes B.
Real 100-update pilot of the first new utility head resumes inside its budget.
Native arm64 CPU4, interop1, workers0; checkpoint/heartbeat every200 updates.
Stop on nonfinite loss, identity mismatch or free disk below10GiB, preserving
checkpoints. Do not stop or reduce the experiment merely because it is slow.

## Eleven Controls Per Pair
Reference; unguarded candidate (diagnostic); neural utility+neural risk; ridge
utility+ridge risk; the two crossed utility/risk combinations; neural and ridge
on common positive-gain support; equal-count neural-risk and ridge-risk ranking;
and a deterministic hash-ranking control at the same count.

All original learned gates use positive predicted gain, the latest-step moving
guard, nonidentical rollouts and the unchanged 0.02 predicted all-harm ratio.
Common support additionally requires both predicted gains and both reference
moments strictly positive. Within each past query, K is the smaller number of
original neural/ridge interventions on common support. Both risk rankings select
K lowest predicted harm/reference ratios with stable row-ID ties. Hash selects
K using a predeclared salt. K includes unknown-label rows, never outcome masks.

This matches intervention COUNT, not a common estimated budget. Each ranked
prefix satisfies its own predicted 0.02 condition; the hash control need not.
Report all predicted and realized risks instead of implying equal scores or a
safety theorem. Report zero-K queries and cases where rankings are identical.

Neural and ridge use the same labels/features within a pair, but neural risk
includes hurdle and ranking losses while ridge uses moment regression. This is
a comparison of complete scoring systems, not an isolated nonlinearity effect.
Full versus motion-only changes the action pair; do not call that a perfectly
matched predictor comparison. The within-pair controls are the matched ones.

## Evaluation and Interpretation
Freeze all396 decisions (18groups x 2pairs x 11policies) before fresh readout.
Reuse all38,102 rows on the same six opened model-selection localities. Include
partial/unknown futures in inference; errors use registered support. Primary
contrasts: neural versus ridge within each pair; neural versus ridge at matched
count; full neural versus retrained motion-only neural. Also compare the current
full neural bridge, training-selected classical forecast, CV, easy/hard/complete
subsets, endpoint, p95/p99, harm and intervention rate.

Use all three seeds and3,000 paired locality bootstrap samples. Average seedwise
gains within locality, not forecasts; six physical localities remain the
independent unit, not396 views or overlapping windows. Intervals are exploratory
and not multiplicity adjusted. Do not pick a best seed or automatically promote
a deployment. Keep adverse results and missing support visible.

Twelve calibration and six confirmation localities remain closed. Eight observed
and twelve requested native annotation steps, image pixels, released detector
tracks. No metric/seconds/physical-safety/human-gold/true3D/foundation claim.
No future input, central velocity, test goals, Stage5C or SMC.
