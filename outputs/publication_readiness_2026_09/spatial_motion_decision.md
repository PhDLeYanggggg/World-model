# Resolution and Spatial Pooling: Fit-Only Repair

The previous observed-motion experiment completed 54 negative neural fits.
The next input hypothesis is specific: a central median may discard a small
moving foreground, while 96-to-32 reduction may remove useful local detail.
Neither hypothesis is yet a measured cause of the real forecasting failure.

## Fixed Design Before Fitting

Decode the same registered past/current frames into masked 96x96 native crops.
Require exact replay of the existing 32x32 image/coverage cache under its known
3x3 reduction. No new query, label, role, coordinate claim or time unit is admitted.
Retain all 11,966 existing fit windows, including missing-video Zara03. Images
are offline annotated observations, not strict real-time sensor evidence.

Compute fixed Farneback flow on two native-size views: original crops and their
32x32 block averages upsampled back to 96x96 by nearest-neighbor replication.
Both estimators use identical parameters: pyramid .5, levels3, window45,
iterations3, poly_n7, sigma1.5, flags0. Restore integer crop translation, use the
same supplied H axes, and keep forward/backward consistency as a proxy only.
This compares retained image resolution at a common estimator grid, not an exact
reproduction of the previous 32-pixel Farneback parameters.

Compare four matched-capacity inputs:

- quality_control: geometry and common spatial coverage/consistency from both views;
- lowpass_pool: add a central median direction and upper magnitude, broadcast to the grid;
- lowpass_grid: add localized mean direction/magnitude in fixed 4x4 cells;
- native_grid: add the same localized representation from original pixels.

All variants share both views' quality fields, the same input dimension, and
the unchanged past-only normalization. Quality therefore contains some motion
reliability information; it is not a pure geometry control. A synthetic sparse
foreground fixture must show that the grid can preserve a vector the median
erases. This test alone does not establish person localization in real images.

Use the already studied row/log objective, the same MLP, three seeds and three
physical-scene held fit folds, final checkpoint at 4,000 updates: 36 fits.
The choice retains the stable log-loss branch while isolating observation/pooling;
this comparison does not claim that raw-ADE training is now repaired. Do not
select a different objective, checkpoint, threshold or variant on held outcomes.
Primary remains past-normalized ADE with equal physical-scene aggregation.

Report raw forecasts, native-coordinate diagnostics per recording, easy harm,
stationary slices, train-vs-held error, and fixed binary oracle. Pair the lowpass
grid versus lowpass pool and native grid versus lowpass grid contrasts. Three
seeds and 2,000 scene bootstrap draws are exploratory on three historically
used scenes, not independent confirmation. Every failed model remains reported.

## Resource and Claim Boundaries

Local arm64 CPU4/interop1/workers0, with optimizer/RNG checkpoints and heartbeat.
85GiB free before conversion; native image cache estimate about 1.1GiB. Keep
arrays, original media, predictions and weights under ignored local data. CREATE
is unnecessary for this bounded experiment; its prior access blocker is not
retested or treated as evidence about current remote jobs.

This is a candidate-quality repair for the baseline-relative joint-intervention
research question, not a new world-model contribution by itself. The preserved
protocol excludes new development/calibration/confirmation access. Stage5C/SMC
stay disabled. No metric/seconds, human-gold, foundation or true-3D claim.
