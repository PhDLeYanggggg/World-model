# Method Addendum: Rejecting a Neural Intervention, Not a Forecast

This is an exploratory controller experiment on fixed neural predictions. It
does not constitute a new dynamics model or an independently confirmed method.

## Forecast and Decision

For indexed agent query i, let N_i be the frozen neural trajectory and D_i the
source-excluded protected damping/CV trajectory. The existing controller gives
a binary intervention s_i. The delivered forecast is

    P_i = s_i N_i + (1 - s_i) D_i.

The proposal only removes interventions. A causal acceptance function A_i gives
s'_i = s_i A_i. It uses the eight observed positions, model-generated trajectories
and fitting-source support boxes. It has no access to future positions, future
validity at inference, zero-reference error labels or held-out statistics.

## Support Representation

Let v_j be the magnitude of successive observed displacements, and v_bar their
mean. The support vector contains latest speed / v_bar, mean observed turn angle
across pairs of nonzero displacements, and mean distance between N_i and D_i
divided by 12 v_bar. These scalar features are invariant to common translation,
rotation and positive isotropic coordinate scaling, as tested numerically.
They are not invariant to temporal resampling, anisotropic transforms, homography,
changing camera motion or missing observations. Relative invariance does not turn
image pixels into verified physical coordinates.

Exact zero denominators have a separate motion state and finite zero feature
values. Fitting-only boxes are conditioned on source and motion state. Membership
in two boxes from distinct fitting sources is an engineering heuristic, not a
probability statement about the new source. The floor and neural producers also
change between the inner cross-fit and outer application. Source-box statistics
cannot remove that producer transport issue by themselves.

## Current-Frame Matched Controls

A query cohort q comprises indexed eligible agents from one recording at one
observed current frame. The quota k_q equals the guard's number of interventions
in that cohort. A risk-ranked control chooses k_q agents with the smallest
predicted harm-to-baseline-risk ratio among original interventions. A random
control uses a fixed hash ordering. Ties are deterministic. Neither rule pools
quotas across recordings, scenes or future frames.

This is not a collision-avoiding joint optimization. In particular, a cohort with
only one eligible intervention has no nontrivial ranking choice. The published
decision-context audit distinguishes fully rejected cohorts, flexible quotas and
actual allocation differences. Identical decisions are retained in the matrix but
cannot count as separate evidence for two mechanisms.

## Error Accounting

Using held-out development labels only after all new decisions are frozen,
define delta_i = ADE(N_i) - ADE(D_i). For removed interventions R, avoided harm
is sum max(delta_i, 0) and lost benefit is sum max(-delta_i, 0). Both are divided
by the same source-specific sum of floor error over the metric's known-label
population. Their difference therefore reproduces the source's change in gain
over the floor. Unknown labels contribute to coverage counts, never zero error.

Source-equal aggregation prevents a large recording population from setting the
whole result. Paired locality bootstrap retains the existing eight-source roster
per fold and repeated seed/fold dependencies. The intervals remain conditional
development uncertainty, not selection-corrected confirmation or risk guarantees.

## Limits of the Hypothesis Test

The stopping rule was motivated by four already inspected partial-label cases.
Its subsequent scores remain exploratory even though implementation and the
comparison matrix were committed before new readout. Improving those cases cannot
repair the independence of the evaluation or establish endpoint safety. The
relevant question is whether causal selection adds value over quota-matched
controls without sacrificing complete-label, hard or worst-source performance.
