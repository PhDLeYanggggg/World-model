# Past-observation-quality Auxiliary

## Material Passport

Researcher-owned code experiment on previously exposed European source data.
Run/validate scope: test incremental past-quality information, not invent a new
forecaster or claim an independently confirmed world model. Raw data stay local;
small learned checkpoints may be stored under the existing owned CREATE root.

## Hypothesis and Controls

The raw-label diagnostic found that complete future supervision contains most
selected harm. Past line-fit residual and dense raw-prefix support showed
descriptive within-query associations, while several other intervals overlapped
zero. This motivates a predictive test, not a noise diagnosis or data filter.

Keep all 72 existing forest cost heads, original branches, train/validation
partitions, targets, risk budget and action thresholds. Add a centered ridge
regression to each original terminal leaf, predicting corrections to its five
cost moments from **all seven fixed past proxies**. The raw conditional mean
correction is zero on the known training rows of every leaf. Therefore this is
not an intercept recalibration or a repetition of global shrinkage.

The information arm uses the actual past proxies. The matched placebo arm
permutes those seven features jointly within each training recording, preserving
recording membership while breaking their within-recording association with
the row. Both arms receive the same frozen tree routing, supervised rows,
weights, original-train normalization, ridge penalty and number of regressions.
Both are evaluated with actual past proxies. Placebo is a diagnostic control,
not a deployment model. The original forest with no auxiliary is the third arm.

## Fitting

Use exactly the prior raw-label diagnostic's seven past features; no future
quality fields or feature selection from the previous confidence intervals.
Compute weighted means/std on known TRAIN rows only, clip standardized features
to [-8,8], and normalize each leaf's positive query weights to sum to one.
Fit slopes by (weighted centered covariance + I)^(-1) times centered cross
covariance. Fixed ridge penalty is 1.0, without validation search. Output units
use the original train-only moment scales. Known training leaf means and weights
must reconstruct the frozen forest. Unknown train outcomes are not zero targets.

Average the per-tree correction, add it to the frozen raw moments, and apply the
existing feasibility projection after nonnegative clipping. The original
envelope, moving mask, radial feature support and zero score thresholds remain
unchanged. This changes a cost predictor, not a trajectory or world rollout.
Record clipping and correction magnitudes; zero-quality slopes must reproduce
the original predictor exactly. Inference accepts past inputs only.

No new tree splits, neural gradient updates, target rescaling by query envelope,
future availability features, sample filtering, relabeling, hyperparameter grid,
independent-role access, Stage5C execution or SMC.

## Readout and Advancement

Primary: quality-minus-original normalized signed-score MSE on the same source
validation recordings. Information-specific control: quality-minus-placebo MSE.
Also report conservative selected-set utility, full and at exactly equal counts
within recording/frame, against each control. Rank matched actions by their own
predicted utility, not observed benefit. Preserve all missing outcomes and their
envelope completion bounds; an empty or undefined risk denominator is not a pass.

Average repeated heads/controllers within locality before 3,000 bootstrap draws.
All intervals are nominal exposed-development evidence, not formal confirmation.
These three head seeds share upstream43, not three end-to-end training replicates.
The registered exploratory advance-to-transfer screen requires both MSE upper
CI bounds <0, both full and matched utility lower CI bounds >0 against both
controls, no fewer complete support passes, no more 2% easy-risk upper violations,
and no larger worst easy-risk upper bound than the original. Otherwise stop this
arm before transfer and retain its failure; do not choose a winning seed/penalty.

## Runtime and Reproduction

First run one real paired fit as pilot, measuring fit/inference, RAM and checkpoint
size. Fit and replay both arms on all72 groups if feasible. Freeze each completed
checkpoint, report and hash. Resume with the same protocol and verify existing
remote bytes, never silently replace them. A verification phase repeats fits,
inference and report arithmetic and writes separate runtime receipts.

Native arm64 local CPU, four threads, num_workers0. Keep the unchanged 10GiB
numeric-cache reserve; current local free space is below it. Store no new local
numeric arrays/checkpoints. Use only the authorized owned CREATE storage root
for checkpoints, with a 1GiB cap; shared free space is not a personal-quota claim.
Any remote scientific training would require the scheduler; none is planned.
Timeout or access failure is not a scientific result. Preserve existing work.

Scope remains obs8/pred12 raw stride12, image-local detector-silver. No physical
identity verification, metric, seconds, human-gold, physical safety, true3D,
foundation, independent confirmation or submission-readiness claim.
