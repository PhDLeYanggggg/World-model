# Matched Harm-Deviance Cost Training

## Motivation And Hypothesis

The completed SOURCE comparison shows selected H/EH underprediction already on
fitting data, not solely on new localities. This narrow training test adapts
existing stable cost-deviance code to the current nine-producer SOURCE cost
framework. It is not a new divergence, newly discovered clipping bug, or claim
that the historical144-fit easy-deviance failure was successful. That old code
already uses stable logs. The new question is whether jointly addressing H
and EH rather than EH alone improves current cost identification and decisions.

## Fixed Comparison

Cached control: original mean_absolute critic. New arms:

- easy_deviance: replace only EH normalized squared error with stable deviance.
- dual_deviance: replace both H and EH normalized squared errors.

For nonnegative cost y, prediction p and frozen SOURCE RMS s, the replacement
is 2*(p/s-y/s+(y/s)*log((y/s)/(p/s))), with zero-target convention0log0=0.
This is a cost Bregman/quasi-likelihood objective, not a claim costs are integer
Poisson observations. Under a sufficiently rich unconstrained conditional model
both terms target conditional means of observed training costs, not unobserved
future outcomes; finite capacity, decoder constraints and
the joint objective can still bias them. Stable log probabilities avoid losing
the nonzero-target gradient when a decoded probability is numerically tiny.
No formal calibration guarantee follows from this loss.

Keep B/R/ER squared terms, absolute gain anchoring, model architecture, parameters,
initial weights, SOURCE normalization, source rows, query sampler, draws,
optimizer, source-only easy cuts, budget and 2% evaluation rules fixed.
Three forecaster folds by actual seeds17/29/43 yield18new fits. Every fitting
query retains at least four visits; no slow-run budget reduction.

All cost fitting uses the producer-excluded eight SOURCE localities. No exposed
selection, calibration, confirmation or DroneCrowd labels fit any parameter or
normalizer. Unknown source outcomes do not receive zero supervision. The causal
model interface does not accept future endpoints, validity, track-remaining
length, central velocity or test-derived goals.

## Runtime And Reproducibility

Native arm64 .venv-pytorch, CPU4/inter-op1, workers0. Save complete model,
optimizer, RNG, sampler/draw hash and update count every100updates, with heartbeats.
Honor10GiB disk reserve and12h healthy-phase boundary; resume unchanged checkpoints.
First perform50->100resume versus100direct for all3arms. The mean control must
match the existing original100step state. Then complete a first-producer mean
control refit and compare original full weights/optimizer/draws/trace exactly.
Refit both first-producer new arms completely and compare full scientific states.
Model weights/caches remain private. Full SOURCE fit diagnostics are optimistic,
not validation, and do not alone pass a deployment gate.

## Readout And Frozen Decision

Freeze every new model, prediction and independent choice before the new
comparison readout. The six selection localities are already exposed development
data; freeze ordering does not make them untouched test data. Use all38,102targets,
28recordings, six localities and nine producers, including unknown/partial future.
Retain the mean_absolute cached control and check its original per-site scores.

Primary: dual_deviance minus mean_absolute FULL paired completion-lower utility.
The easy-only arm is a controlled factor, not another primary to choose afterward.
Use3,000locality bootstrap draws after repeated producer/seed averaging. Native
known utility, missing-outcome bounds, full-record positive all/easy harm and
easy-net requirements remain distinct. Empty reference is not a safe zero.
Primary superiority also requires a positive lower CI; deployment additionally
requires every original recording view to meet the original risk contract.
Report both arms and all failures, not just the most favorable mean.

A separate complete replay recomputes18new model inferences, choices and54site
metric groups. These are controlled development results, not independent final
confirmation or a population-risk certificate. Do not fit another multiplier
or repeatedly tune on the exposed calibration cohort in this experiment.

## Unchanged Claim Boundaries

Obs8/pred12/rawstride12, local image pixels, detector/tracker silver, available-
label ADE rather than full12 ADE. Unknown-row bounds do not recover missing
steps inside partially labeled trajectories. Endpoint FDE is diagnostic where
missing. No metric/seconds, human gold, true3D, foundation, physical-safety or
new trajectory-dynamics claim. No Stage5C execution or SMC. Confirmation stays
closed. A successful training run is not paper readiness or guaranteed acceptance.
