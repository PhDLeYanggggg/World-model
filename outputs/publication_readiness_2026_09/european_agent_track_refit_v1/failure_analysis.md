# Result Interpretation and Remaining Failures

## Supported, Modest Topology Effect

The fixed topology-only contrast passes its exploratory forecaster screen:
ADE gain vs matched flat is0.4465%, CI[0.1559%,0.7480%], and hard ADE gain
is0.5024%, CI[0.2515%,0.8163%]. All nine producer/seed point gains are positive.
Seed17's interval still crosses zero[-0.0058%,0.7968%]; do not describe all
three seeds as individually significant. Ten locality point gains are positive,
two negative; locality020 is worst at-0.589%. FDE gain is0.4615%, with
interval[0.0711%,0.8686%]. The effect is small, not a large-model breakthrough.

The primary controls use identical partial geometry, initial parameter values,
88,514 parameters, loss, baseline choices and final sampler states/counts.
All9 control predictions reproduce exactly. This rules out those measured
execution mismatches as explanations of the contrast. It does not isolate
temporal pooling from attention factorization or prove social reasoning.

## Comparator Choice Matters

Against the older complete-neighbor neural bank, ADE gain is0.459% but the
interval[-0.033%,0.936%] crosses zero. Do not combine two successive positive
point estimates into a confirmed gain over the original architecture.
Against the training-selected causal reference, ADE gain is3.388%, interval
[1.381%,5.604%]. Against CV, the all-query gain3.126% has interval
[-7.922%,10.214%]. The latter is not stable domination of the strongest floor.

Easy ADE degrades0.161% vs the matched flat control, within the registered
2% *neural-comparator* guard. Against CV it degrades13.080%, with gain interval
[-26.252%,-3.263%]. This fails a deployable easy-preservation claim. The small
forecaster benefit cannot be renamed policy safety; no risk policy was trained.

## Association Response Is Now Present

For each of nine models, an input-only probe reassigns two complete neighbor
tracks at selected past times while preserving point sets at every time.
Grouped outputs respond: mean changes range0.00396 to0.57782 source-coordinate
units. Flat means range5.18e-7 to4.17e-6. Eight flat models meet the fixed
approximate-invariance tolerance; the ninth has maximum rounding-scale change
0.000366 and fails that strict numerical tolerance. Do not call all flat
outputs bitwise invariant. No future labels or accuracy enter this probe.

The multiple-neighbor slice has0.556% gain[0.351%,0.806%]; partial-history gain
is0.529%[0.312%,0.780%]. No/one-neighbor slices lack complete fixed-roster
support, so their aggregate intervals remain undefined. The slices overlap
and are not retrained ablations. We cannot yet attribute the whole gain to
neighbor interaction rather than improved ego encoding or pooling.

## Output Range Is a Limitation, Not the Whole Explanation

The label-only projection diagnostic finds43.48% of valid target steps outside
the existing causal correction disk (equal-locality average). The geometric
lower-bound error is23.13% of grouped model error on average across locality
views; the hard counterpart is27.28%. This is an oracle lower bound, not a
learned prediction. Stationary-motion misses remain in the evaluation.

Yet mean budget use is only7.31% overall and10.68% on hard. Fewer than0.002%
of overall valid nonzero-budget steps use more than95% of the radius on the
same locality-average basis. This argues against describing the model as
generally stuck at its cap. Relaxing the bound is not justified by these numbers,
especially while easy error is already unacceptable. These are descriptive
ratios averaged across dependent producer/seed views, not additive error shares.

## A Separate Coordinate-Unit Sensitivity

All72 input-rescaling cases fail approximate scale equivariance after restoring
output units. Baselines themselves pass. No future labels are read. For the
grouped bank, mean returned-coordinate changes range0.455 to17.141; the largest
single coordinate change is423.029. These are source-local coordinates, not meters.

The composition restores the observed context scale before applying the radial
squash, then multiplies by a motion budget that already supplies units. Even
with identical normalized core inputs, changing coordinate units changes the
correction fraction. The probe deliberately avoids the one-unit input clamp,
so that clamp cannot explain these cases. This is directly measured input
sensitivity, not proof of the cause of held error or a promised repair gain.

A separate dimensionless-fraction candidate now removes ONLY this pre-squash
scale restoration, keeping the grouped core, budget and parameter shapes.
Its synthetic tests check units while the unchanged input clamp is inactive,
initial floor, finite gradients, missing-neighbor robustness and bound preservation.
It is UNTRAINED and not part of the current nine-model result. A matched fresh
experiment is required; silently swapping it into trained checkpoints would
invalidate both the contrast and learned calibration.

## Scientific Boundary

The source set is development-exposed. Locality bootstrap with overlapping
producer fits is exploratory, not independent confirmation. No new threshold,
scene-joint policy, calibration or deployment was selected. Results remain
obs8/pred12 raw-stride12 image-local silver-label evidence, not t+50, calibrated
seconds, metric, true3D, foundation or submission-ready world-model evidence.
