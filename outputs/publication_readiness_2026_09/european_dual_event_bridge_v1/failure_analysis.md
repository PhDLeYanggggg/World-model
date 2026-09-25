# Failure Analysis: Aligned References Help, Predicted Caps Are Not Guarantees

## 1. Why the Old Score Combination Was Invalid
The original easy and all policies could deliver different fallback forecasts.
Their risk scores therefore referred to different prediction changes. This
study aligns the complete R-to-P decision before making utility, all-risk and
easy-risk labels. Producer A remains excluded from controller B's labels, and
the new controller sees real past-history/neighbor/rollout features.

The resulting all-risk-only bridge is better and easy-stable on the opened
selection cohort. This supports this implementation of aligned policy-level
cost learning; it does not isolate reference alignment from all accompanying
feature and action-pair changes as a unique causal explanation.

## 2. Why Dual Risk Is Not the Best Result
The extra easy-risk condition vetoes 176,703 repeated row/role/seed actions
accepted by all-risk-only. In all 108 dependent locality/views, the rejected
actions have more aggregate benefit than harm, so the dual veto raises total
ADE. This is exact accounting, not an oracle used to deploy a new policy.

Dual risk reduces easy positive-harm exposure, but sacrifices much of the gain:
all-ADE gains vs old easy add_only shrink from 3.95%-5.57% to 0.00061%-1.78%.
Five of six paired seed-mean CIs favor easy-risk-only over dual risk. The
additional all constraint within that conservative subset has no demonstrated
accuracy contribution. It cannot be advertised as a successful dual-risk method.

## 3. Why Easy Preservation Is Not Safety Certification
The 2% easy-degradation rule is a net error difference against CV. Positive harm
against R deliberately does not cancel harmed agents with benefited agents.
All-risk-only passes net easy in all 18 views but exceeds the easy positive-harm
ratio in 68/108 locality/views. Dual risk still exceeds it in three. This is
calibration error and event-support uncertainty, not an excuse to redefine the
event or replace positive harm by net improvement after looking at results.

Thirty of 36 risk heads reduce their fixed training-batch loss; six do not.
Training loss, even when it decreases, is not evidence that conditional moments
are calibrated on a new locality. No risk bound is currently validated on the
unopened calibration population.

## 4. What Produced the Gain
Across repeated group/seed decisions, all-risk-only makes 175,971 motion-to-motion
switches with a pooled ADE-reduction sum of 624,221.50 pixel-error units,
63,404 switches to neural forecasts with +75,725.30, and 4,659 switches away
from neural forecasts with -494.53. These repeated pooled sums are diagnostic
only: they are not the equal-locality primary metric or independent samples.

The dominant effect is selecting between existing motion forecasts. It is not
proof that the neural backbone learned a better dynamics law, or that scene
images, goals or interaction tokens caused the gain. A matched motion-only
policy bridge, matched single-risk ridge control and controlled intervention
rate are the shortest next checks before stronger neural-method claims.

## 5. Remaining Scientific Risks
Six readout localities have been used for model selection and must never become
confirmation. The experimental family shares those sites and frozen forecasts;
reported CIs are conditional and not multiplicity adjusted. Source groups have
unequal row counts and model performance varies by producer/controller. Detector
tracks, partial futures and incomplete-neighbor encoding limit interpretation.
No threshold, risk tolerance, checkpoint or readout population was changed after
seeing these results. No calibration/confirmation outcomes were opened.
