# Source-Only Component Calibration Control

Registration precedes fitting any new margin. This is a follow-up diagnostic
within the already exposed European development pool. No global endpoint,
prediction protocol, scientific success threshold or independent data role changes.
The parent crossed-head experiment is frozen and remains a negative risk result.

## Hypothesis and Fixed Comparisons

Head-seed changes did not remove risk failures. Offline decomposition found
both harm underestimation and reference overestimation. Test three fixed
component controls: harm-only upper adjustment, reference-only lower adjustment,
and joint adjustment. Retain the unchanged raw/source-screened policy, floor,
and a per-query count-matched joint-versus-parent comparison. The primary
diagnostic is same-count ADE utility; risk, coverage, hard/easy utility, unknown
outcomes and undefined denominators are reported together. No best-arm, seed,
quantile or threshold selection using transfer outcomes. All three head seeds
remain, with upstream43 fixed. New neural/forest training is not part of this run.

## Source Recording Calibration

Use the same72 frozen heads, each with its original source optimization/validation
partition. Heads and preprocessing were fitted only on optimization recordings.
Source validation recordings were not used to fit these fixed-budget forests.
Define raw eligible rows using past-only model scores and existing causal support.
For each whole source-validation recording, aggregate residuals on known raw
eligible rows. Scores, in order, are:

1. `sum(actual_H - predicted_H) / sum(causal_max_disagreement)`.
2. `sum(actual_easy_H - predicted_easy_H) / sum(causal_max_disagreement)`.
3. `sum(predicted_R - actual_R) / sum(predicted_R)`.
4. `sum(predicted_easy_R - actual_easy_R) / sum(predicted_easy_R)`.

Zero denominators yield missing scores, not zeros. Unknown labels are excluded
from residual fitting but recorded and retained in source risk completion bounds.
Each recording supplies one score per component, irrespective of its window
count. Fit the fixed empirical90th percentile using the higher order statistic;
clip each nonnegative adjustment to[0,1]. A missing component's support disables
the calibrated policy for that source. A single contributing recording is enough
to compute a diagnostic coefficient but does not establish uncertainty coverage.

Leave one entire validation recording out at a time. Fit margins on all other
validation recordings, apply them causally to the held recording, and pool these
OOF decisions for the unchanged finite-completion source screen. Known masks
must never alter per-row inference eligibility. Use all source-validation
recordings to fit final transfer coefficients. Also report full-fit source
resubstitution screens separately; they are not the selection criterion. OOF
coefficients differ from final coefficients, so an OOF screen is not a finite
guarantee for the final transferred policy.

## Adjustment and Decision

Harm margins add the fitted fraction of causal maximum disagreement, capped by
that disagreement; all-harm upper is at least easy-harm upper. Reference margins
multiply predicted reference by one minus the fitted fraction; easy reference
lower is at most all reference lower. Keep benefit prediction unchanged, except
disable utility when source support or either predicted denominator is absent.
Do not apply a joint moment projection that could shrink a harm upper estimate.
These adjusted decision scores are not claimed to be coherent expected moments.

Use the original2% selected-reference budget and positive predicted utility.
Calibrated raw actions must be subsets of original raw actions. Source screening
can change which sources are admitted, so screened actions need not be subsets
of the parent's screened actions. Match intervention counts within each query,
rank by the corresponding causal utility, and break ties with fixed row IDs.

This is empirical component calibration, not a conformal implementation or a
cross-domain/physical-safety guarantee. The90th percentile is not a change of
the2% risk budget or an advertised90% coverage certificate. Quantile intervals
and selected expected-harm risk are different objects. Report whether any
apparent safety gain merely removes all useful interventions.

## Execution, Checks and Claims

Pilot the first real source calibration locally. CPU4/interOp1/workers0, native
arm64, no new large row cache or model weights. Preserve10GiB disk reserve.
Save each source calibration for resume and PID heartbeat. Commit all72 source
calibrators/screens before216 transfer-head action views; commit actions before
reading their outcomes. Exactly replay all72 calibrations and the entire readout.
Reconstruct native metrics, match query counts, and require all648 old raw,
source-screened and floor metric comparisons to agree exactly with the parent.

Report all policies, all heads and3000 nominal locality-bootstrap draws over12
already exposed localities. Repeated windows and folds are not independent
samples. No deployment promotion follows from all-zero actions, undefined risks,
small development gains or choosing a favorable seed. Independent selection,
calibration and confirmation stay closed. Obs8/pred12, stride12 raw frames,
image-local detector-silver only: no metric, seconds, true3D, foundation,
human-gold or physical-safety claim. Stage5C execution and SMC stay off.
