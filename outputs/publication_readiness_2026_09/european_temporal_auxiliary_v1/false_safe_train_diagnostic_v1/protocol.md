# Frozen-Head TRAIN False-Safe Diagnostic

## Scope

Following the failed fixed seven-arm development readout, diagnose all 216 final
neural cost heads on their original 24 TRAIN packets. This is resubstitution,
not a new validation result, independent confirmation, or deployment claim.
No weights, targets, split roles, budgets, thresholds, or original code change.
There are zero optimizer updates. The original temporal experiment stays failed.

The source loader is used locally only to check the original TRAIN movement mask
against the frozen causal CV rollout columns 355:379. Every TRAIN row ID and
feature hash must match the frozen training receipts. Failure stops execution;
neither an all-moving assumption nor a label-derived movement mask is allowed.
The loader accesses the already exposed source arrays; it does not create new
validation predictions or access independent calibration/confirmation roles.

## Predetermined Readout

Reuse the exact causal eligibility rule: moving, supported, predicted gain >0,
predicted all-harm excess <=0, predicted easy-harm excess <=0, at the frozen 2%
budget. Unknown outcomes remain eligible; labels are used only in diagnosis.

Decompose selected known-row realized easy-risk excess as:

```
actual_harm - .02 * actual_reference
  = predicted_harm - .02 * predicted_reference
  + (actual_harm - predicted_harm)
  + .02 * (predicted_reference - actual_reference)
```

Report signed component masses, rather than only positive errors, normalized by
the original TRAIN cost scale. Report known/all and known/selected cohorts,
unknown selected counts, causal support, per-query and per-recording violation
counts, undefined zero-reference cases, and fixed quartiles. Label substitutions
are mechanism-only oracle diagnostics and cannot become inference features.
Repeated predictions must match exactly. Independently sum every decomposition.
Do not infer scene-level safety from a per-row false-safe count. Source/seed
views overlap and are not independent samples.

## Execution And Limits

One owned CREATE CPU allocation: 4 threads, interop 1, workers 0, 16 GiB, 2 hours.
Read the already verified 24 source packets and 216 final checkpoints in place.
Record ownership, hashes, job ID, heartbeat and aggregate-only output. No new
checkpoint, feature array, raw data or remote environment installation.
Retain the 10 GiB storage reserve and do not touch other projects or jobs.

The result can choose a subsequent bounded TRAIN-only repair hypothesis. It may
not tune the completed validation readout, open independent roles, claim a
generalization gain, or relax safety. Detector-silver image-local raw-frame
coordinates remain uncalibrated; no metric/seconds/true-3D/foundation claim.
Stage5C and SMC remain off.
