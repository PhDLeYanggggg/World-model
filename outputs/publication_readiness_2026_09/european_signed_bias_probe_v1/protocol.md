# Fitting-Only Signed-Risk Intercept Probe

## Material Passport
- An analytic two-parameter fit per frozen risk head, not new neural training.
- Development follow-up to the frozen-action diagnostic registered in b2ab3ae6.
- Parent neural heads, forecasts, floor, utility and actions remain unchanged.

## Question and Fixed Procedure

The frozen diagnostic finds positive mean residual even on fitting rows.
Row-weighted residual is not the query/source/subset-weighted fitting objective.
Before changing the policy, determine whether an additive score intercept can
improve the *exact original fitting objective*, or whether weighting explains
that apparent bias. Do not select a deployment threshold from held outcomes.

For each of108 frozen role groups and both heads, predict scores on the two
fitting sources only. Hold model parameters fixed. Use the same three causal
proxy banks, known-label support, equal source sampling and equal known-query
sampling as the parent fit. Compute the full empirical objective, not a
stochastic monitor minibatch. Retain the0.5individual anchor and0.5subset term.

For an additive signed-score offset d, each arm's empirical quadratic has the
same derivative structure: mass*d - weighted_truth_minus_prediction. Empty
subsets contribute zero, including their quadratic mass. Fit the exact
nonnegative minimizer d=max(weighted_residual/mass,0), one scalar per all/easy
axis. Report the unconstrained solution too. Nonnegative offsets cannot admit
an action rejected by the old pointwise risk rule. Do not apply offsets to
the deployed or development actions in this probe.

Cross-check the full objective difference against the analytic quadratic
identity and automatic differentiation on synthetic examples. Report offsets,
losses, weight mass and source/seed variability for all216 heads. Save fitted
coefficients locally with identities and hashes; push only aggregate results.

This probe tests optimization/weighting, not generalization, uncertainty or
risk calibration. Training loss reduction is guaranteed by the analytic fit
and therefore cannot count as a learned downstream success. No held future
labels enter this fit; no new held evaluation or threshold search is run.
Any later policy experiment requires a separate preregistration and frozen
actions, a same-count control and the unchanged risk tolerance.

## Boundaries

Same twelve development localities and4/4/2/2role contract. Independent selection,
calibration and confirmation stay closed. Obs8/pred12 raw-frame stride12,
image-local detector silver; no metric/seconds/true3D/foundation/human-gold claims.
No deployment, Stage5C or SMC. Native arm64 CPU4/interop1/workers0. Group-level
checkpoint/resume; retain10GiB free. No large data or checkpoints in Git.
