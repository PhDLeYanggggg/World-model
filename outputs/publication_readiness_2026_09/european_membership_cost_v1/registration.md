# Membership-Conditional Expected Harm

## Material Passport

Previous goal turn: progress. Direct easy-membership MLP passed its registered
full-input diagnostic, including a stronger training-prevalence sensitivity;
expected-harm and deployment gains remained unproven. Parent d9891a87 is
cached_verified, including 56 public artifacts and 96 source bindings. This
round fits fresh cost heads, not new forecasts or a policy. Training has not
run at registration. Independent selection/calibration/confirmation stay closed.

## Hypothesis

Let E = 1(0 < CV_error <= fitting positive-error 25th percentile), H be positive
candidate-minus-reference error and e the causal forecast-disagreement bound.
The inherited event includes harmless easy rows and excludes exact-zero CV
error. Unknown labels are excluded, not made negative. Both H and E are
supervision/evaluation only. No future input, central velocity, held endpoint
goals or held normalization.

Test whether separating membership from within-membership harm improves
expected easy-harm accuracy. This is a conditional-expectation decomposition,
not a novel estimator by itself and not proof of a scene-level decision rule.
The approved forecasting endpoint, role allocation and 2% risk tolerance stay
unchanged.

## Models and Controls

Two new 383 -> 64 GELU -> 2 MLPs, each 24,706 parameters:

1. Direct: H_all_hat = e sigmoid(a), H_easy_hat = H_all_hat sigmoid(b).
   Fit mean squared errors of all/easy harm, normalized by fitting target RMS.
2. Conditional: m_E = e sigmoid(a), m_notE = e sigmoid(b). Fit H on the E and
   not-E strata, including zero harms, with each conditional squared loss
   normalized by fitting conditional RMS and stratum prevalence. The common
   equal-locality sampler is unchanged; no class oversampling.

At inference only, compose the conditional heads with the frozen previous
MLP p(E|x): H_easy_hat = p m_E and H_all_hat = p m_E + (1-p) m_notE.
This keeps 0 <= H_easy_hat <= H_all_hat <= e. Reference-cost predictions remain
bit-identical to the original mean head. Both cost heads use the same causal
input features; no realized E/H label enters prediction.

The membership probability is NOT an input to, or part of the loss of, either
new head. It is only used for post-fit composition. Thus no in-sample stacked
probability is fitted as a second-stage feature. Training composed-cost
diagnostics use the in-sample frozen probability and are optimistic fitting
diagnostics only; held predictions use the three-locality membership model,
which excludes the held locality. Any future stacked-input model must use
nested/out-of-fold probabilities.

Retain original_mean as a cached control. Also compose the SAME conditional
experts with constant training-only p(E), without retraining or thresholds.
This tests whether learned membership, rather than just conditional training,
contributes. All variants are reported; no best variant is chosen on held data.

New cost heads share initial encoder weights, dimensions, sampler draws,
2,000-update budgets and optimizer settings. Their output priors and loss
scales differ by target, so initial losses are NOT matched. Conditional also
uses an extra 24,641-parameter frozen membership model and its prior training.
The comparison is equal new-head budget, NOT equal total system capacity/cost.
The direct arm's failure cannot by itself establish architectural novelty.

## Roles, Runtime and Freeze

Six ordered source assignments x three seeds (17/29/43) x full/motion-only
forecast pairs x four held localities x two new arms = 288 fits / 576,000
updates. Source-A producers exclude B. Three B localities define all fitting
normalizers, event cuts and priors; the fourth is excluded throughout fitting.
These are previously exposed source-development localities, not independent
confirmation. Six selection, twelve reserved calibration and six confirmation
localities remain unused. Context loaders may read cached held arrays but
held targets enter neither fitting nor selection.

AdamW lr0.0003, weight_decay0.0001, batch256, clip5; CPU4/interop1/workers0 in
native arm64. Checkpoint and heartbeat every200 updates, exact resume. First
100-update direct pilot counts within the fixed budget, then resume all fits.
Report conditional timing separately. No time-based downscaling, no new HPC
job; local pilot/resources decide placement. Keep10GiB disk reserve. A fresh
read-only CREATE queue check leaves existing jobs untouched and does not
verify an M3W remote artifact directory. Preserve private raw data/weights.

Commit registration before fitting, and commit/push prediction hashes before
this round's held-label readout. No favorable-role dropping, early stopping,
post-readout fitting, new thresholds or policy readout.

## Registered Readout

Report all and positive-disagreement easy-harm MSE, full moment MSE, positive-
harm AUROC/AP, top10 harm mass capture, coverage and four membership/harm
error partitions. All bins use fitting predictions only. Retain full and
motion-only results, fit/held gaps and every role/seed/locality.

Primary cost signal requires full positive-disagreement easy-harm MSE gains
with positive95% intervals in all six source assignments against BOTH direct
and original_mean, and against the conditional constant-probability control.
Additionally, top10 capture, absolute log-coverage error and all-row H_all MSE
must have no negative or non-estimable interval against direct/original_mean.
These guards are not non-inferiority or safety guarantees. Missing support
cannot pass. Passing permits only consideration of a later registered policy
test, not deployment or access to independent data in this round.

Average three seeds within locality, then3,000 resamples of four localities
per assignment with seed47131. Roles/windows overlap and are dependent; CIs
are exploratory, not multiplicity-adjusted or independent replication. Full
and motion-only disagreement populations differ, not a causal modality ablation.

Detector-derived image pixels, eight observed/twelve predicted annotation
steps. No human-gold, metric/seconds, physical-safety, true3D, foundation or
submission-ready claim. Stage5C and SMC remain off.
