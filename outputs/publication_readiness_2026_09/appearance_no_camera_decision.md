# Matched Retraining Without Learned Camera Jacobian Inputs

## Material Passport

Exploratory fit-only single-factor repair after the registered feature-box
controls. None of those controls produces positive guarded gain. Strict box
fallback rejects all cross-scene rows. Same-original-mask decomposition separates
prediction changes from confidence/gate changes; some apparent harm reduction
is abstention, not better forecasts. This is not a new primary experiment.

## One Changed Factor

Retrain the eight-past-RGB arm with columns28:32 of the normalized32-dimensional
geometry input set identically to zero in both training and inference. These
four supplied-H camera Jacobian components had large cross-scene shifts. Keep
the original28 causal geometry/neighbor features, all image/missingness inputs,
normalization, model dimensions, initialization seed, sampler, targets, losses,
optimizer, update budget and0.9 diagnostic gate unchanged. The deterministic
image-displacement-to-native-coordinate output mapping still uses supplied H;
the experiment does not remove coordinate conversion or add physical calibration.

Use all365 original stationary fit rows. Fold0 holds ETH; fold1 holds Hotel.
Three seeds17/29/43, two folds =6 new models at1,000 updates each. Compare each
with its hash-verified original past-RGB checkpoint and report all six pairs.
No outcome selects a seed, threshold, update or feature set. These exposed fit
sites cannot become independent confirmation. Full benchmark/calibration/final
test roles remain untouched. Source native-index timing remains an assumption.

## Execution and Claims

Native arm64 CPU4/interop1/workers0. Save optimizer, RNG, losses, PID/heartbeat,
checkpoint every100 updates and per-trial predictions/receipts. Resume partial
fits exactly, reuse completed receipts only after hash verification. The prior
real runtime already shows local execution costs a few minutes, so no HPC job
or additional resource probing is needed.

Report unrestricted and guarded native/parent-normalized ADE/FDE, start Brier,
easy absolute harm, switch rate, parameter/update/compute counts and all seeds.
Easy percentage remains undefined atzeroCVfloor. Better than the damaged
original model is not better than CV; report both comparisons. If this repair
remains nonpositive, do not keep tuning thresholds or claim that dropping all
camera/scene context is generally justified. The conclusion concerns only this
input component in this tiny fit-only predictor.

No retrospective change to the eight-observed/twelve-predicted parent metric,
raw50 supplement, source-use terms or pending new primary-metric decision.
No Stage5C, SMC, new deployment, physical safety or submission-readiness claim.
