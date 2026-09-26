# Severity-Weighted Auxiliary Supervision

## Material Passport

Parent 27a8c53b completed the frozen optimizer diagnosis. It did not support
broad total-cost gradient conflict: 5/72 full auxiliary views had negative
cosines, while 68/72 disposable auxiliary steps improved total cost more than
cost-only. All-harm components remained mixed and the prior held-cost gate
failed. This is one target-alignment repair, not a gradient/weight sweep.
Source forecasts and matched controls are cached_verified; support, new
training and new source-held readout will be fresh_run after registration.

## Hypothesis

Ordinary membership BCE estimates P(E | X). The nested harm fraction instead
represents q(X)=E[H E | X]/E[H | X] where E[H | X]>0. They need not agree.
A Bernoulli loss weighted by nonnegative H has q as its population optimum:
its conditional expectation is -E[H E|X]log(q)-E[H(1-E)|X]log(1-q).
This elementary identity is not a new algorithm, calibration guarantee, or
claim that q is learnable from the available causal features.

Keep the exact four nested cost MSEs and their fitting RMS normalization.
Replace only auxiliary BCE by mean[(H / fitting_mean_H) * BCE(logit,E)].
Coefficient remains1. The fitting mean uses the original equal-locality
known-row weights. No weight clipping, resampling, class balancing, task
weight selection or threshold search. Loss weights and labels are detached
supervision, never inference inputs. The auxiliary output is not multiplied
into predicted costs and is no longer an ordinary membership probability.

Keep architecture, initialization (including ordinary-membership bias),
optimizer, sampler, fixed batches, updates and inference identical. Do not
silently initialize at a new prior. This isolates one loss change. The four
cost outputs remain nested; only H_all/H_easy replace original readouts,
while D_all/D_easy stay frozen as before. Plain harm-only regression already
failed and is not being rerun under a new name.

## Fitting Support Before Training

Compute training-only positive mass/counts separately for E and not-E at
window, recording-agent track, recording and locality levels. Sum mass within
groups before reporting Kish-style concentration. This is NOT an estimate of
independent sample size: overlapping windows and recordings remain dependent.
Report ordinary easy prevalence and harm-weighted fraction separately.

Training is numerically allowed only if every view has positive all-harm mean
and all full-input views have positive harm mass in both E strata. An absent
stratum is not repaired with held labels. Flag fewer than20 positive tracks
or fewer than2 supported localities in either stratum as weak, without
discarding favorable/unfavorable folds or asserting statistical power.
Retain all motion-only support failures as limitations. No missing support
may be labeled a successful generalization result.

## Budget, Roles and Freeze

Six assignments x three seeds17/29/43 x full/motion-only x four held localities
=144 new heads /288,000 updates. Width64,24,901 parameters; AdamW lr0.0003,
weight decay0.0001, batch256, clip5, 2,000 steps; checkpoint/heartbeat every200.
Run a real100-step pilot included in its budget, then resume. Native arm64
CPU4/interop1/workers0. Preserve10GiB reserve; choose CREATE only if justified
by measured resources, do not alter current jobs or restart slow runs.

Three fitting localities determine all preprocessing, cost scales, means,
labels and training weights. The fourth and source-A producer chain remain
excluded from fitting. Use original source rosters; previously exposed source
development is not independent test. Six selection,12 reserved calibration
and6 confirmation localities stay closed. Freeze/push prediction hashes
before the new source-held outcome readout. No outcome-based model choice.

## Comparators and Criterion

Compare severity_aux with original_mean, matched cost_only and ordinary
membership_aux. All cached comparators are hash/schema/row checked. Require
all six full positive-disagreement easy-harm MSE intervals positive against
all three, plus no negative/not-estimable top10 capture, coverage-log-error
or all-row all-harm MSE contrasts. Failure against a strong control cannot
be rescued by beating a previously failed model or one selected assignment.
This is the existing component diagnostic endpoint, not a new forecast or
deployment gate. Report motion-only and adverse results fully.

Three seeds averaged per locality;3,000 resamples of four localities per
assignment, seed47131. Source roles/windows overlap; these exploratory CIs
are not independent confirmation or multiplicity-adjusted. Report fitting/
held transport, loss components, support concentration and compute costs.

## Boundaries

Detector-derived pixels and8 observed/12 predicted annotation steps only.
No future endpoint, central velocity or held endpoint goals at inference;
future errors only fitting/evaluation supervision. No metric/seconds,
human-gold, physical-safety, true3D, foundation, deployment or submission-ready
claim. Stage5C/SMC off. This experiment cannot establish the overall project
goal alone. All prior negative, contaminated and exploratory evidence remains.
