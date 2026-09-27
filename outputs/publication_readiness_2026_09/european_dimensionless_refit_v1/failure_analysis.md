# Improved Forecasts, Unresolved Intervention Safety

27 September2026. Fresh source-development readout, not independent confirmation.
The comparison was registered in9f090d6a; predictions were frozen in c686ae06
before scoring. All nine models and all prespecified views are retained.

## 1. The Parameterization Repair Has Measured Benefit

Matched grouped-control ADE gain is4.7209%, with a3,000-locality-bootstrap
interval of[2.5308%,7.8854%]. FDE gain is6.4922% [3.3758%,11.0994%]. All
nine producer/seed point gains and all three seed intervals are positive.
Eleven of twelve locality ADE means improve; locality119 loses0.4527%.

Hard ADE improves3.8739% [2.5654%,5.0828%] versus grouped control. Against
the training-selected causal reference, all-ADE improves7.9161%
[5.3633%,12.0571%]. Against CV, all-ADE improves8.4616%
[1.9118%,12.9795%]. These are mean locality percentage gains, not pooled
coordinate errors. This passes the registered predictor-development screen.

## 2. Easy Preservation Remains a Deployment Blocker

Mean positive-easy ADE gain versus grouped control is0.9502%, but its interval
[-0.7069%,3.3551%] crosses zero. Relative to CV, easy gain is-11.1692%
[-21.2576%,-3.1438%], so error remains materially worse than CV.

The mean-only matched screen does not protect every locality. Easy gain vs
grouped control is-4.1239% at locality082. Locality020 improves19.4622%
against the old grouped model, yet still loses20.7867% all-ADE and52.4939%
easy-ADE against CV. Improvement over a poor neural control is not a safe floor.
No adverse locality is dropped or used to select a new inference exception.

## 3. Reference-Exact Queries Get Worse

The four zero-CV-error queries in locality008 appear in six dependent
producer/seed views, not24 independent queries. All six views show a larger
mean error than the grouped control; all four rows worsen within each view.
New mean ADE ranges0.6097 to1.5497 image-local units; grouped means range
0.1304 to0.5568. These changes are not rounding noise. Percentage gain vs
zero CV remains undefined. The complete absolute-cost tables retain the values.

The causal motion budget does not identify all reference-exact future outcomes.
The exactly stationary-history floor is a narrower guarantee. Never identify
these cases at inference using their future error or add a label-based exemption.

## 4. Unit Sensitivity Is Repaired Within Its Stated Support

On the identical observed-history prefixes, all36 new scale checks pass the
old numerical tolerance, whereas all36 grouped checks fail. Factors are
0.25/0.5/2/4 and the conditioning clamp is inactive. No future labels enter this
probe. Across the full input cache,317,587 histories support all factors;
700 of318,969 have an active clamp in original coordinates. No query was
removed from training or scoring. Arbitrary tiny-unit, rotation, camera or
temporal-stride invariance remains unproved.

## 5. Optimization Changes Alongside Unit Consistency

All729 recorded control gradients exceed the fixed clipping threshold5;
only3 of729 candidate records do. Recorded median norms change from roughly
85--231 to0.93--2.63. This samples81 logged steps per trial, not all updates.
The initial losses are identical and sampler states/counts match.

Removing scale restoration changes optimization as well as the mathematical
unit property. The experiment supports the combined parameterization repair;
it cannot causally isolate "equivariance" from gradient scaling or effective
residual amplitudes. A gradient diagnostic is not a new independent experiment.

## 6. Representation and Calibration Claims Remain Open

Multiple-neighbor and partial-history slices have positive matched intervals,
but no-neighbor/one-neighbor full-roster intervals are undefined for insufficient
support. These descriptive slices are not retrained interaction ablations.
No new image, goal, JEPA, scene-joint intervention or risk-head contribution is
tested. The source localities and earlier outputs were development-exposed;
three seeds and a bootstrap cannot turn them into independent confirmation.

## Consequence

Freeze this improved forecast bank as a development candidate. The highest-value
next experiment is matched gain/harm learning and intervention with this bank,
including equally protected simple-motion controls, rather than another round
of unrelated architecture expansion. Refit the cost producers/heads with full
producer-chain exclusions; do not reuse old head calibration as if candidate
cost distributions had not changed. Easy and zero-reference harm are mandatory
readouts. Only then can independent calibration and confirmation be justified.

Obs8/pred12, raw-frame stride12, detector-derived silver image trajectories.
This8.46% all-ADE number is not the historical Stage37 raw-t+50 number.
No deployment change, Stage5C, SMC, metric/seconds, human-gold, true3D,
foundation or submission-ready claim.
