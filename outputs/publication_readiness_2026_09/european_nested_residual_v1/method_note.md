# Nested Residual Supervision

## Question

Does fitting a fixed easy-harm correction to locality-out-of-fold risk
predictions transport better than fitting it to training-locality predictions?
This does not test a new trajectory forecaster or a deployed selector.

## Roles and Targets

Each outer risk-estimation view contains three fitting localities and one
excluded source-development locality. Trajectory producers were trained on
their separate producer roster. For each of the three fitting localities,
a new risk head excludes that locality and the outer locality. Its feature
standardization, target scales, easy cut, sampler and initialization use only
its two remaining fitting localities.

The predicted moments remain D, H, D_E and H_E: reference cost, positive
candidate harm, easy-subset reference cost and easy-subset harm. Future
outcomes define supervised targets only; the risk-head inputs are past
features and frozen causal rollouts. Unknown labels have zero fitting weight.
The easy event is learned from the inner fitting cut. It is not replaced with
an outer-fitting cut that includes the inner-held locality's labels.

## Matched Residual Banks

For fitting localities ordered a, b, c, the OOF bank uses the head excluding
the row's locality. The two fixed cyclic controls use the head excluding the
next or previous locality. Each has one producer per row, trained on two
localities with the same architecture, seed and update budget. Both cyclic
controls include the row's locality during head training; OOF does not.

This matches training-set locality count, not its composition. Inner label
cuts also differ; all three banks use their producer's own easy target and
report disagreement with the outer-fitting cut. Consequently, a difference
between banks cannot identify exposure bias in isolation from composition
and cut transport. The unchanged original outer estimator and the previous
three-locality in-sample correction are also retained.

## Fixed Correction

Seven causal summaries are path efficiency, speed change, mean turn angle,
current neighbor count, nearest-neighbor distance, closing speed and rollout
disagreement. Coordinates and velocities are normalized using observed path
length, current width and observed speed. Missing neighbor values receive an
explicit missing bin; they are not imputed from future trajectories.

The global probe fits an intercept; the context probe adds indicators for
fitting-weighted terciles and missingness in each summary. Both fit the
normalized residual `(H_E - predicted_H_E) / fitting_RMS(H_E)` with a fixed
ridge penalty of 0.1, excluding the intercept from the penalty. Only known
rows with a positive causal displacement envelope enter this residual fit.

At readout, every probe corrects the same frozen original outer estimator:

```text
corrected_H_E = clip(original_H_E + fitting_RMS * context_shift,
                     lower=0, upper=original_H)
```

D, H and D_E are unchanged. The bound is an algebraic moment constraint,
not a physical-safety guarantee. Fixed ridge and features are not selected
using held readouts. Corrections are not applied to trajectory outputs.

## Interpretation

There are 144 dependent views and six producer/controller assignments, with
three seeds. The reported bootstrap averages seeds within locality and
resamples four localities per assignment 3,000 times. Four localities are
limited support; overlapping assignments are not independent replications.
The registered primary comparison requires the full-input OOF context repair
to improve against all five controls across all six assignments, with the
registered tail/coverage guard. Motion-only and other contrasts remain
reported even when unfavorable. A failed gate does not authorize opening
independent selection, calibration or confirmation roles.

The study uses eight observed and twelve predicted annotation steps in
image pixels. It supports no metric, seconds-level, human-gold, physical-safety,
true-3D, foundation or submission-readiness claim.
