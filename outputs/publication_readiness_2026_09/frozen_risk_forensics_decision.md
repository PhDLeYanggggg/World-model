# Frozen Risk Forensics

Specified 2026-09-20 after the complete frozen-coupling comparison. This is
post-hoc descriptive diagnosis on already opened development labels, not a
new calibration or confirmatory experiment. The primary-metric decision remains
pending and is not adopted by this work.

## Question And Fixed Scope

Why do all new controls satisfy their predicted query-harm constraints but fail
the separate easy-degradation requirement? Use all 24 completed predictor/head/
policy combinations and all three new controls, with no new inference, training,
threshold selection, primary change, or unseen data. Verify parent hashes,
receipts, row identities, label coverage and aggregate comparisons first.

Separate three quantities:

1. Frozen predicted mean positive harm over all past-eligible query agents.
2. Realized positive harm of the labeled selected agents divided by that same
   original agent count. This is only a lower bound under nonnegative completion
   of unavailable selected costs, not a zero imputation or a new population claim.
3. The existing relative net ADE increase on the complete-label easy cohort,
   where easy is defined by the original future baseline-error threshold.

Unswitched agents have exactly zero excess by construction, even if labels are
unavailable. Missing labels on a switched agent remain unknown conservatively,
including cases where the forecasts might coincidentally be equal. A query with
such an unknown cannot be declared within the realized budget merely because
its observed lower bound is low. If the lower bound already exceeds the budget,
the query demonstrably fails it under every nonnegative completion. If no
selected cost is missing, evaluate the budget exactly on this stored population.

## Planned Diagnostics

- Count predicted-budget failures, proven realized-budget failures, completely
  scored within-budget queries, and indeterminate queries.
- Measure observed positive harm, easy positive harm and signed easy excess in
  each budget-status group. Retain all missing-label decisions and denominators.
- Report predicted/observed query-harm reliability in fixed quarter-budget bins,
  separately identifying fully scored selected-agent queries. Expected cost is
  not a failure probability, so do not label these ECE or probability calibration.
- Compare head/family/policy strata without selecting a new winner.
- Reconstruct the existing easy metric from the rows and require agreement with
  the frozen report. Report both absolute excess and relative percentages.
- Include a constructed truthful-prediction counterexample: a global average
  absolute-harm budget need not imply the conditional relative easy constraint.
  This is an algebraic nonimplication, not proof of a novel method.

No independent uncertainty can be inferred from the single physical development
site. Repeated windows, policies and seeds are not new sites. Do not compute a
window-bootstrap interval or infer whether a training defect versus scene shift
caused underprediction from this descriptive evidence alone.

## Boundaries

The results may refine the failure diagnosis and clarify what a future training
and calibration objective must address. They do not authorize a new loss, metric,
split, threshold or deployment. No test-driven repair is performed. All source
artifacts remain untouched; only code and aggregate reports may be committed.
Stage5C and SMC stay disabled. No metric, seconds, true-3D, foundation or physical
safety claim. Dataset-provided offline annotations remain distinct from strict
sensor-as-of histories.
