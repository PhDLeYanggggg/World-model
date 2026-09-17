# Numerical Follow-Up, Not a New Forecasting Hypothesis

The registered conditional-ADE experiment completed all18 settings. It preserved
expected training risk and exposed23/39,420 waypoint solvers that reached the
5,000-iteration limit before the strict objective-gap tolerance. Do not call
these converged. The first results remain preserved, not overwritten.

Refine only those23 flagged waypoints using the SAME opposite-fit-scene training
labels and leaf weights. No held outcome selects the indices or solver setting.
Eight have an exact support-atom solution under the convex subgradient test;
this was diagnosed from training support only. Check every support atom, then
use analytic-gradient BFGS on the remaining smooth two-dimensional objective.
Retain the1e-8 support-scale gap tolerance. Optimizer success is not sufficient:
the subgradient/hull certificate determines numerical convergence.

Recompute all18 fixed scores and compare original versus refined predictions.
Do not change the original forests, features,0.9 gate, target, primary metric,
data roles or candidate selection. Never promote a favorable numerical change.
Any remaining uncertified solve is explicit, not silently hidden by fallback.
No neural training, new site, calibration/confirmation access, deployment,
metric/seconds claim, Stage5C or SMC is authorized by this numerical repair.
