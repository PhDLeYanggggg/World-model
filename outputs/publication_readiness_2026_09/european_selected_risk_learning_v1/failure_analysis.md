# Failure Analysis

## The Objective Fits Better Than It Transfers
The selected-group penalty lowers its fixed-batch error in 26/36 paired fits,
but all-ADE selected-joint versus mean-joint is negative at the point estimate
in every source assignment, for both full and motion-only forecasts. Five of
six intervals per pair exclude zero negatively. This rejects the proposed
loss as a reliable accuracy improvement at this architecture and budget.
It does not prove that all subgroup-learning methods fail.

The penalty constrains three B-derived subsets per training locality, including
the old raw selections. The new policy selects a different subset. Matching
those finite means need not calibrate the newly selected population. B's old
teacher selections are in-sample. These are plausible mechanisms, not causes
isolated by this experiment. Longer fitting, cross-fitting and a different
objective have not been tested here and must not be claimed as fixes.

## Joint Decisions Buy Coverage But Spend Uncertain Budgets
Full selected-dual intervenes on 6.58%-26.74% of indexed rows; selected-joint
intervenes on 23.06%-41.20%. Joint decisions beat independent decisions in all
six three-seed accuracy contrasts, but complete risk passes drop from 17/18
to 9/18. The matched-count control gives four positive and two inconclusive
intervals, so the improvement is not established as a universal ranking gain.
The budgets are predictions, not observed future error or a certificate.

## Net Easy Error Conceals Positive Harm
Full selected-joint's 14 easy-event violations occur at localities 074 (6),
126 (3), 112 (2), 119 (2) and 020 (1), among 72 dependent locality views.
These repeated views are not 14 independent failures.

For fold1/seed29/controller2 at locality074, net easy error improves 3.4930%
relative to CV, while easy-event positive harm relative to delivered R is
5.6957%, above the fixed 2% bound. All-event positive harm is 1.4005%.
The comparisons have different denominators and should not be conflated.
Improvements on some examples can cancel deterioration on others in the net
metric; they cannot cancel the positive-harm numerator.

## Residual Optimism and Label Support
For full selected-joint, predicted selected harm is below actual harm in
46/72 locality views; the median predicted/actual ratio is 0.8453. Mean-joint
has 43/72 and 0.8911. Different actions prevent interpreting these numbers
as a controlled calibration-only effect.

Queries include all indexed agents, including those whose future labels are
unknown. This is required to avoid using future availability at inference;
metrics use supported labels only. Consequently predicted mass from unknown
futures can finance known-row actions while being absent from the observed
denominator. The [posthoc budget accounting](query_budget_audit.md) checks this
mismatch without changing decisions. It must never become a future-support input.
In the worst locality, unknown rows supply only 1.4143% of predicted easy mass,
while supported selected easy harm is predicted as 184.75 versus an actual
1,862.20. Conditional numerator underprediction, not unknown budget alone, is
the main measured discrepancy there. Its modeling cause remains unisolated.

## Remaining Attribution Gaps
- Motion-only selected-dual passes every observed risk setting, but it is
  markedly less accurate than its old raw rule. Do not rename abstention as a
  learned dynamics breakthrough.
- The old ridge control passes 12/18 complete settings on each pair. New neural
  costs have not established robust dominance over simple controls.
- No new forecaster was trained; this round cannot establish a Transformer,
  JEPA, interaction-consistency or neural trajectory contribution.
- Four C localities per assignment, historical development exposure and
  overlapping roles prevent an independent generalization or risk certificate.

The deployment and reserved-data state remain unchanged. Image-pixel,
annotation-step, detector-derived results only. No metric/seconds, human-gold,
physical-safety, true3D or foundation claim; Stage5C and SMC stay off.
