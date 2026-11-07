# Learning When To Depart From OLS4

Test whether a learned modulation of motion increments beats a
matched ungated correction. This is a deterministic supervised
forecaster, not latent generation or a replacement for the
external selector. Use the existing 69,746 past-eligible fitting
rows (nine nonempty sources) only. All 14 Moving A development
source groups are excluded from fitting and preprocessing.
DroneCrowd calibration and confirmation are not used.

Both arms estimate OLS4 velocity from the same past points, build a
local heading frame and normalize by root-mean-square past speed
with a fixed pixel floor. From seven observed displacement pairs,
a two-layer MLP predicts three bounded longitudinal and lateral
velocity increments per axis. Interpolate those knots over 12
steps and integrate from the last observation. In the gated arm
only, two learned sigmoid factors modulate the entire correction
along and across heading. Both arms have the same initialized
weights, parameter count, source-balanced sampling, mask-aware
past-scale-normalized ADE objective and 900 updates at batch size
128 for each of three seeds. No validation or test checkpoint
selection. Use the final frozen update for each arm.

Freeze all six past-only development forecasts before scoring.
Compare all-row proposals to CV and OLS4 and compare the two heads.
For the principal operational contrast, substitute the forecasts
only for the original matched gain policy's recipients, keeping
the same path when either alternative would be identity to CV
within one millionth of a pixel. Both heads therefore use the same
substitution set and the same original count in every query. Keep
original support, 2% recording risk, partially observed labels,
and missing-label utility. Report source-paired 3,000-draw intervals
after averaging seeds, easy degradation and tail ADE. Positive
evidence needs a conservative improvement over the matched
ungated arm and against the original fixed-recipient gain policy;
no result can change deployment without independent evaluation.

Run natively on arm64 CPU, two Torch threads, no data loader workers.
Keep atomic checkpoint and heartbeat for resuming each fit. Use
eight observed and twelve forecast annotation steps in image pixels.
No metrics in meters or seconds, no Stage5C, no SMC.
