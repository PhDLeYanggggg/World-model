# Signed Budget-Excess Model and Data Card

## Purpose and Parameterization

This is a fixed-budget diagnostic change to the controller objective, not a new
trajectory forecaster.144 native Torch heads retain the exact64-wide network,
22914parameters, source normalizers, AdamW settings, gradient cap, three seeds
and2000updates of the sealed two-moment controls. Every sampler/draw/RNG state
is compared with its control; fresh control inference is checked against cache.

The two-output architecture is retained to isolate the objective change. Its
new components are NOT separately identified reference and harm moments. Only
the signed combination `q=p1-0.02*p0` is supervised. A large component canceled
by the other is not evidence of calibrated individual expected errors. The old
moment outputs and new basis outputs must not share that interpretation.

The fixed q<=0 screen is not a learned probability, not a conformal bound and
not the earlier complete utility/easy deployment rule. The positive-harm ratio
does not subtract successful interventions and is not net ADE degradation.

## Data and Exclusions

Only the twelve already opened European source-training localities. The frozen
forecast producer uses four localities outside the controller roster. Each
head fits three of the four controller localities and is checked on the fourth.
The outer readout is not scored. Independent model-selection, calibration and
confirmation roles remain closed. These are development-exposed sources.

Inputs:355past-only target/neighbor geometry and baseline/candidate rollout
features. Future labels/masks construct supervised errors only, never features.
No future endpoint, target latent, central velocity, test endpoint goals or
held-source normalization. Unknown paired labels are excluded, zero-reference
positive harm retained. Easy/hard cutoffs were fitted by the frozen producer.

Forecasts/geometry are cached_verified, heads fresh_run. Detector-derived silver
image-local tracks, obs8/pred12 at raw-frame stride12. No human-gold, metric,
seconds-level, physical-safety, true3D or foundation claim. Overlapping windows
and shared producers are not independent; twelve-locality bootstrap is exploratory.

## Release and Limitations

Weights, row predictions and data remain private/local and Git-ignored. Public
artifacts contain code, fixed configuration, hashes and aggregate diagnostics.
Exact cached replay is not a cold reconstruction from raw data or proof of
independent safety. A calibration policy requires nested exclusion of the whole
producer chain. No deployment change, Stage5C execution, SMC or submission.
