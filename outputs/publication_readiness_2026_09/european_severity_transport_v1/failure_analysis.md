# Failure Analysis: What the Frozen Diagnostic Can Establish

## Mechanism Ledger

| Proposed explanation | Evidence | Conclusion and limitation |
|---|---|---|
| A few held recordings carry the additional error | 24/38 worsening full/original views have >=50% positive excess mass in one recording | Descriptive concentration is present; recording size, target mass and overlapping windows can also explain concentration. Not causal attribution. |
| One fitting recording usually dominates weighted auxiliary gradients | 11/72 full views and 26/72 motion-only views reach 50% | Registered broad trigger fails. Weighting raises median concentration relative to ordinary BCE, but this final fixed batch cannot characterize training-path influence. |
| Held failures mostly lie outside the fitting feature radius | Enrichment >20pp in 10/38 full/original and 2/44 motion-only worsening views | Registered broad trigger fails. Coarse marginal radius cannot rule out conditional shift or unsupported local patterns. |
| Weighted membership is sufficient to repair cost magnitude | Parent full MSE: no positive intervals vs original or ordinary auxiliary; tail/coverage guards failed | Rejected by the completed matched experiment. Event AUROC improvements do not establish expected-cost accuracy. |
| More training or a larger network is the demonstrated remedy | Neither is intervened on here | Not established. This diagnostic is not justification for an indiscriminate architecture or loss-weight sweep. |

## Important Denominators

Each pair contains 72 views from the same source-development system, with
overlapping assignments and repeated seeds. The 24/38 statistic refers to
worsening full/original views, while 11/72 refers to all full fitting views.
These are not independent Bernoulli trials and are not causal probabilities.
Full held views contain 3-62 recordings (median 4); the fixed fitting batches
contain 11-52 recordings (median 27.5). Concentration thresholds are not
adjusted for different group sizes or row exposure. The group counts are a
post hoc description of the retained artifacts, not a new selection rule.

Easy-harm MSE here is a cost-head prediction error, not trajectory FDE and
not an observed deployment easy-degradation percentage. Raw positive excess
and negative excess must both be kept: their difference gives signed error,
but cancelling them before concentration accounting would conceal damage.
Mass ESS is a concentration summary, not an effective independent sample size.

## What Remains Unresolved

- Full-path gradient variance and optimizer influence are unmeasured; only
  one fixed diagnostic fitting batch per frozen model is considered.
- Similar radial magnitude does not imply similar conditional features or
  target distributions. No density-ratio or conditional-support proof exists.
- Within-recording target noise, detector errors and omitted causal context
  have not been separated by this calculation.
- Some fit improvements still fail to transfer. The current results do not
  identify a loss, sampler, architecture or missing feature that fixes that.
- The original, ordinary auxiliary and weighted auxiliary results remain
  development evidence, not an independent forecast or policy evaluation.

## Next Repair Criterion

Keep all source assignments and the original strong control. Define a small
set of motion/interaction contexts using fitting features only, then ask
whether cost residual direction and magnitude recur when localities are
held out. This must not use realized future/easy labels to define an inference
feature, delete difficult contexts, or retrofit an independent test role.

A repeatable causal-context bias would motivate a single context-specific
cost-model repair. Inconsistent or sparsely supported patterns would instead
motivate an explicit information/support limitation, not favorable threshold
selection. The repair and its expected outcome remain not_run. This report
does not reverse the failed parent cost gate or change deployment.

Detector-derived pixel coordinates and annotation steps remain the evidence
units. No metric/seconds, human-gold, physical-safety, true-3D or foundation
claim. Stage5C/SMC remain disabled.
