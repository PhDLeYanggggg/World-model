# Model and Data Card: Static-Gradient Loss Control

## Intended Use

Research diagnosis of candidate utility before baseline-relative intervention.
Not for deployment, autonomous control, physical-safety certification, metric
forecasting or a general-purpose world-model claim.

## Model

Twelve new deterministic SourceDynamics models, 44,864 parameters each; four
source-site folds and seeds 17/29/43. Geometry and coverage inputs use the
existing mask-only arm; RGB tensors are zeroed. This experiment does not train
a new JEPA/Transformer architecture, language model or generative rollout.
Past-derived support/radius constrain complete relative trajectory outputs.

Training uses 10,000 updates, batch size 64, AdamW learning rate 0.0003 and decay
0.0001, gradient clip 5, 2,000 constant-rate and 8,000 cosine updates, and final
learning-rate ratio 0.01.
Only entirely zero-target rows' ADE gradients are removed. Full batch
denominator and all sample exposure remain. Terminal checkpoints are fixed;
no held-score selection or routing threshold is learned.

## Data

SDD source material admitted by the existing hash-bound data-role contract.
15,430 stationary-history queries at coupa/deathCircle/gates/hyang, 29 recordings,
545 recording-scoped agents; 8,566 entirely zero-target and 6,864 nonzero-target
rows. Queries overlap. Eight observed/twelve predicted annotation steps at
stride 12 raw frames. Inputs are past-indexed offline annotations, not verified
real-time sensing. Coordinates are annotation pixels, normalized from past
information; time-to-seconds and metric scale remain unverified.

Each normalizer, loss scale and training-hard cutoff excludes its held physical
site. Bookstore is excluded throughout; main and sealed evaluation roles remain
unscored. Existing assets may be loaded for identity checks, but excluded roles
are not used for fitting or forecasting. Future targets are loss/eval labels
only. There are no test-endpoint goals or central-velocity official inputs.

## Evidence and Limits

New fits/analysis are fresh_run; prior unconditional controls are cached_verified.
Raw held ADE gain is -98.71920% equal-site versus stationary CV. Candidate/CV
future oracle rises to 3.75968%; rotated controls retain 3.31027%-3.58218%.
Zero-target harm is 1.42880 annotation pixels. Easy percentage is undefined.
These are conditional development estimates, not independent validation.

All 12 forecasts replay exactly; matched exposure and 68-file immutable resume
pass; 34 focused tests pass. Full legacy suite was not rerun because it can
execute training and rewrite historical reports. No checkpoint, per-row cache,
dataset, video or image is included in the public Git commit.

The model has no verified causal routing, multimodal contribution, physical
validity, cross-dataset gain or deployable safety guarantee. Existing historical
deployment candidates are not re-certified here. No Stage5C execution or SMC.
