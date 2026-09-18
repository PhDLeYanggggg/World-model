# Training-Side Candidate Cross-Fitting: Pre-Fit Decision

Date: 2026-09-18. This is a diagnostic repair inside the admitted source
training role, not a change to the main split, metric, risk budget or final test.

## Hypothesis and Scope

The fixed deferral readout failed despite a small in-sample gain. Before fitting
another risk head, generate gain/harm supervision from genuinely out-of-site
candidate forecasts. Test whether useful candidate trajectories survive this
exclusion. Cost-target optimism is a hypothesis, not a proven root cause.

The outer source complement is unchanged: 15,430 complete stationary-history
queries from coupa, deathCircle, gates and hyang. Bookstore is excluded from
all fitting and inference; no main data are predicted. Each of the four source
sites is held internally once. Learned feature normalization, cost scale and
hard cutoff use only that fold's other three sites. Recording/scoped-track
aliases must not cross roles. Inner held labels are evaluation/supervision
only, never passed to the candidate trainer or inference inputs.

## Fixed Matrix

Twelve SourceDynamics models: four folds times seeds 17/29/43. Same mask-only
geometry/coverage schema, bounded twelve-step output and ADE objective as the
previous dense controls. All models start from random initialization. Old
bookstore-excluded parents saw every inner held row and are **not reusable**.

Each new model receives 2,000 updates at learning rate 0.0003, then 8,000 cosine
continuation updates to 0.01 of that rate. Batch 64, AdamW decay 0.0001, gradient
clip 5; CPU4, interop1, workers0. Atomic checkpoint/heartbeat every200updates;
resume optimizer, sampler and Torch RNG. Total120,000 updates. A 100-update
pilot in coupa_seed17 counts within the budget. No early-stop or checkpoint
selection; final10,000 only. No new risk head, policy or threshold search here.

Use existing verified fitting engines rather than rewriting optimization.
This isolates provenance repair, not a new architecture. All twelve final
train/inner-held forecasts are retained. Build one aligned OOF prediction and
signed-gain label per outer-training row and seed; export them privately with
fold-training cost scales and producer identities.

## Analysis Frozen Before Training

Primary source diagnostic: improvement in equal-physical-site mean
past-normalized ADE versus stationary CV over these four inner sites. Also
report window-weighted gain, per-site/per-seed scores, hard slice, zero-target
absolute harm, native annotation-pixel ADE, tail errors, and binary future-oracle
headroom. Future oracle is diagnostic, never an inference policy.

Use2,000 fixed-seed38113 bootstrap draws over four site means and separately
over recordings within each site. These explored sites and overlapping training
folds do not yield independent confirmation; intervals are conditional evidence.
Errors are averaged across seeds, not predicted trajectories.

Compare OOF cost labels descriptively with cached, hash-verified full-four-site
in-sample dense forecasts on exactly the same rows. Differences conflate less
training data, site shift and preprocessing changes; do not claim a uniquely
identified training-optimism mechanism. No seed/site is selected from results.

## Resource and Integrity Boundary

Local assets verified; native arm64 Torch2.12.0,65.96GiB free. Existing parents
overlap all4250/3054/1662/6464 held rows respectively, in allthree seeds.
GitHub main matches999f448a. CREATE current assets/jobs remain unverified;
historical access blockers are not current scheduler evidence. Local prior
throughput suggests roughly1-2hours; pilot will measure actual cost before the
full invocation. Slow progress is not failure and is not a reason to shorten
the registered budget. No duplicate remote job is launched.

Only aggregate reports/code/configs are public. Caches, row-level labels,
predictions and checkpoints remain private. Source observation is offline
supplied-history8-to-12 atstride12/+144rawframes. No metric, seconds, true3D,
foundation, human-gold, calibration or deployment claim. Stage5C and SMC remain
off. Main research gates and submission readiness remain unmet.
