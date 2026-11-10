# Arbitrary-path exclusion certificate

Extract the archive and run `python -I -B -S verify.py`. It checks
member hashes and rebuilds the OLS4 upper bound on each of the 72
exposed MovingDroneCrowd++ recording-seed views. It uses the standard
library and does not need tracks, images, weights or a solver.

Each observed step can improve its CV-anchored error by at most the
smaller of its CV error and the actor's peak edit radius. At a missing
step the conservative OLS4 comparison can recover at most the smaller
of its CV-to-OLS4 disagreement and that radius. Adding those concave
piecewise-linear gains gives at most twelve marginal slopes per actor.
Within a query, all actors carry the same weight. The payload contains
only the sum of interval lengths at each slope, an aggregate peak
budget, and recording error masses. Query order is removed.

For segment lengths L with marginal slopes s, the allocation
constraint is total spent length at most B. For any nonnegative
multiplier q, the gain is at most q B + sum(max(s - q, 0) L).
Filling the largest slopes first and taking q at the last boundary
gives a matching bound. The verifier checks both calculations. It
uses the same full observed CV error mass to express gain, and
subtracts the archived comparison's known OLS4 advantage and missing
disagreement at the CV starting point.

This is a necessary condition on the conservative comparison under
the unchanged total peak budget, before count, risk or path-menu
constraints are added. A nonpositive upper bound excludes a dual pass;
a positive one does not certify any deployable proposal. The certificate
replays numerical evidence on development data, not a new evaluation
or a claim about unlabeled positions.
