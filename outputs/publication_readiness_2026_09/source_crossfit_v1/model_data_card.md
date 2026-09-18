# Model and Data Card: Source Candidate Cross-Fitting

## Intended Use

Training-side research diagnostic and provenance-correct candidate-cost labels.
Not a deployable predictor, selector or calibrated safety system. No model from
this experiment is promoted over the baseline. Historical selector scores retain
their exploratory status rather than being reinstated as independent evidence.

## Model

Twelve SourceDynamics models, each 44,864 parameters, initialized randomly with
seeds 17/29/43 in each of four folds. A small coverage/geometry encoder predicts
twelve bounded two-dimensional relative offsets restored through an observed
frame. RGB is zeroed in this fixed mask-only control. Unsupported restoration
frames return the zero-displacement baseline. Bounding and finite outputs alone
do not prove collision avoidance, kinematic validity or physical safety.

Training uses ADE normalized by a training-only cost scale, 10,000 updates per
model, batch 64, AdamW, fixed terminal checkpoints and no held-score selection.
The first 2,000 updates use constant learning rate, followed by 8,000 cosine
updates. No risk head, stochastic proposal, latent rollout, correction deployment
or SMC is trained or executed in this study.

## Data and Roles

15,430 admitted SDD stationary-history rows from coupa, deathCircle, gates and
hyang; 29 recordings and 545 recording-scoped agents. Windows overlap and sites
were previously explored. Each model fits three sites and predicts the fourth.
Bookstore is excluded from all fits and forecasts, including producers and
preprocessing. Main protocol roles remain unchanged and unscored. Verified
shared assets may be loaded by dataset construction; exclusion refers to use.

Eight observed steps and twelve target steps at stride 12 raw annotation frames.
Inputs use supplied historical annotations and observed coverage/geometry.
Future trajectories form losses and cost labels only. Offline annotations may
have been interpolated using later control points; past-indexed access and
loaded-label poisoning do not establish strict sensor-as-of availability.

Native coordinates are annotation pixels; the primary cost uses past-normalized
coordinates. No validated physical time, scale, homography or human-gold labels
are introduced. These claims do not extend to the complete SDD cohort or other
datasets. Data, caches and model weights stay local and are not committed.

## Evaluation and Limits

Equal-site mean normalized ADE gain -5.01598%; window-weighted -5.45692%.
All twelve held-site fits lose to stationary CV. Conditional four-site 95%
interval [-8.39655%, -2.48773%]. No independent calibration or confirmation.
Absolute harm on 8,566 zero-target queries is 0.09190 pixels, with undefined
percentage degradation. Fixed binary oracle has only +0.46765% equal-site
headroom in a post-hoc diagnostic, not an inference policy or general bound.

Reproducibility: twelve exact training/held forecast replays, verified producer
and normalizer lineage, exact OOF coverage and immutable zero-update resume.
See [reproduction](reproducibility.md) and [nested validation limits](method_and_limits.md).
The model is not a verified world-model contribution or submission-ready result.
