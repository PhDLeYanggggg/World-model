# What Can a Small Number of Localities Establish?

This is a sensitivity calculation, not a certificate for the measured ADE risk.
No new confidence level, risk tolerance or independent-data access is adopted.
The accompanying JSON enumerates the registered locality counts and deltas.

## Zero Observed Violating Localities
For hypothetical iid Bernoulli locality violations, zero events in n localities
gives the exact one-sided upper bound `1 - delta**(1/n)` on violation probability.
At delta=0.05 the bounds for n=4, 6 and 12 are approximately 52.71%, 39.30% and
22.09%. This is a probability of a violating new locality, not mean positive
ADE harm divided by reference error. Correlated localities violate the premise.

## Best-Case Bounded-Loss Test
For a hypothetical loss in [0,1], zero empirical loss and a 0.02 target, the
Hoeffding-Bentkus p-value has best-case value `0.98**n`. At n=12 it is about
0.7847, not below 0.05. Reaching 0.05 even in this idealized case needs at least
149 iid localities, before allowing for nonzero risk or multiple decisions.
This calculation follows the bounded-loss testing construction in
[Learn then Test](https://arxiv.org/abs/2110.01052), not a fitted-score guarantee.

The actual positive-harm/reference-error ratios are not established as bounded
in [0,1]. Clipping them would change the estimand and cannot be silently used
to obtain a theorem. Overlapping rows and repeated source/seed views do not
increase the number of independent localities. A parametric model, alternative
estimand or dependence assumption would require a separately justified design.

## Consequence
The source-C maps measure empirical transport. Bootstrap describes uncertainty
within the opened locality sample; neither step establishes a distribution-free
2% deployment guarantee. Keep reserved calibration and confirmation closed until
the policy, estimand and inferential feasibility are adequate. Reporting a
limitation is not permission to relax the target or promote a failed controller.
