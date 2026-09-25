# Conditional Easy-Harm Exposure, Unchanged Expected Objective

## Material Passport
Fresh training-source audit completed before this registration; new fits and
source-C readout have not run. Parent forecasts, moment controls, utility and
features are cached_verified. The six opened selection localities are not
evaluated. Reserved calibration and confirmation stay closed. Historical C
development exposure is not erased by this registration.

## Why This Repair, Not Another Architecture
The preceding controlled experiment improved query allocation over individual
gates but failed conditional harm constraints. The worst locality predicts
about one tenth of actual selected easy harm. Missing-future budget mass is
only 1.4% there and is not an adequate explanation.

The new B-only audit finds 4.75-8.51 positive easy-harm examples per 256-row
batch for the full pair, and 0.65-2.11 for the motion-only pair. Across B
locality views, the top 1% of full-pair examples account for a median 89.65%
of easy harm; the motion-only median is 100%. These are dependent row/seed
views, not independent event counts. The full-pair median fitted/actual easy
mass is 0.7784 on B itself. This supports testing exposure, not assuming it
will fix transport. Event-factorized heads and generic oversampling have
already been tested elsewhere in this project; they are not new contributions.

## Single Changed Factor
Retain the exact coherent four-output EventMomentHead, width64, initialization,
B-only normalizers and component RMS scales, mean-MSE objective, optimizer,
2,000 updates, batch256 and fixed loss-check batch. Reuse the preceding mean
heads as hash-verified controls. Do not use the unsuccessful selected-group
penalty. Only the training draw distribution changes.

Let p be the existing equal-locality supported-row training mass. Within each
B locality, q mixes 50% of p with 50% mass proportional to the supervised
easy-harm target. If a locality has no such mass, q=p there. Every supported
example retains q>=p/2 and its loss weight is p/q<=2. No weight clipping or
self-normalization. Thus E_q[(p/q)L]=E_p[L]; exact full-B loss and output-gradient
checks precede fitting. This identity does not guarantee lower variance,
unbiased clipped Adam updates, convergence, calibration or improved accuracy.
Future labels affect B sampling/loss only, never inference inputs.

The pretrained controller/forecaster chain and source roles are unchanged:
A produces forecasts, B trains this head, C excludes both fitted chains.
All six A/B assignments, seeds17/29/43 and full/motion-only pairs are retained.
36 fresh heads, 72,000 total updates; a 100-step real pilot resumes inside
the first head's budget. There is no test-based early stopping or best seed.

## Frozen Readout
Compare corrected all/dual/scene/joint rules with the corresponding cached
mean rules. Retain R, raw neural, raw ridge, selected-dual and selected-joint.
Add a query-count-matched hash control for corrected-joint. Freeze and push all
504 decision views before C outcomes. No threshold or risk-tolerance changes.

Primary: corrected-joint versus mean-joint observed conditional risk and all
ADE; also compare corresponding individual rules to distinguish fitting from
allocation. Compare corrected-joint against raw neural/ridge, independent dual,
and matched counts. Report all/easy/hard/complete ADE, FDE, tails, interventions,
all/easy positive harm, known/unknown label accounting and moment calibration.
Three-seed locality means, 3,000 locality-bootstrap resamples; four C localities
per assignment, overlapping views, no multiplicity-adjusted discovery claim.

If source fit improves but held-source harm does not, report exposure as
insufficient for transport. If accuracy is lost, retain that tradeoff. Do not
promote a policy just because one seed or source assignment passes. Positive
C findings still require separately frozen transport and independent evidence.

## Resources and Prior Work
Native arm64 CPU4, interop1, workers0, 10GiB reserve, lock/PID/heartbeat and atomic
optimizer/RNG checkpoints. Current CREATE queue has been checked read-only;
no jobs are submitted or modified. Remote M3W asset inventory remains not_run.

Importance weighting is established prior work, not a new method claim.
[Katharopoulos and Fleuret, ICML2018](https://proceedings.mlr.press/v80/katharopoulos18a/katharopoulos18a.pdf)
discuss unbiased reweighted gradients and variance-oriented sampling. The fixed
label-mass mixture here is not their gradient-bound algorithm and inherits no
speedup or calibration guarantee.

Obs8/pred12 annotation steps at raw stride12; image pixels and detector-derived
labels. No metric, seconds, human-gold, physical-safety, true3D or foundation
claim. No new dynamics training, Stage5C execution, SMC or deployment promotion.
