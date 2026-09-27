# Development Ablation: Fitting-Only Risk Centering

## Method and Falsifiable Contrast

We keep the forecaster, protected motion policy, utility predictor and query
eligibility fixed. For each risk axis j (all and easy), the head's normalized
signed score is q_ij = (predicted_harm_ij - 0.02 predicted_reference_ij) / c,
where c is fitted on the head-training sources. A preceding training-only
quadratic fit produced a nonnegative constant offset d_j for each frozen head.
This experiment does not refit those offsets.

The centered independent anchor admits eligible agents satisfying q_ij+d_j<=0
for both axes. If its cardinality in a recording/frame query is k, the centered
joint policy maximizes the unchanged predicted utility subject to exactly k
interventions and sum_i a_i(q_ij+d_j)<=0. The matched control uses exactly the
same k and utility, but constrains sum_i a_i q_ij<=0. Both can fall back to the
same feasible independent anchor. The experiments report both head families.

At fixed k, the new constraint is simply sum_i a_i q_ij<=-k*d_j. Thus a uniform
offset tightens every query's budget in proportion to cardinality; it does not
learn which agent's risk estimate is wrong. This algebra motivates an exact
count-matched control rather than comparing only with the less-abstaining
original policy. It is not a new risk guarantee or a claimed novel theorem.

## Development Result

Across twelve previously opened development localities, centered aggregate
risk worsens ADE by0.00523%against the same-count control (nominal95%paired
locality interval:0.00043%to0.01182%worsening). The pointwise family also worsens
ADE. Aggregate fixed-denominator positive harm decreases0.00101percentage
points, but lost benefit is larger. Relative to its original policy, aggregate
intervention collapses from7.80%to0.67%, with200/216dependent views abstaining
entirely. Six other views still violate the unchanged2%selected-risk screen.

This falsifies the proposed fitting-only global-offset repair on this
development design. It does not establish independent generalization, prove
all conditional repairs ineffective, or convert abstention into calibrated
risk control. The negative result distinguishes objective centering, policy
allocation, coverage and observed safety instead of treating them as synonyms.

## Reporting Boundary

Action generation preceded outcome readout through a committed freeze.
Forecast seeds17/29/43 and overlapping source-role views are averaged within
locality before3,000bootstrap draws. Intervals are nominal and exploratory.
The observation protocol is8past/12future steps at raw-frame stride12 with
image-local detector-silver annotations. Independent model-selection,
calibration and confirmation data remain closed. No metric, seconds-level,
physical-safety, true3D, foundation, deployment or submission-readiness claim.
Stage5C and SMC remain disabled.
