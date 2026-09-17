# Frozen Candidate Ceiling Diagnostic

Post-hoc diagnostic on already exposed fit folds, not a preregistered accuracy
test. No new predictor, selector or threshold is fitted. Reuse all 72 fixed
track/event-sampling checkpoints/predictions, after hash and row-alignment checks.
All eight condition combinations, three seeds and three physical-scene folds
are included; no condition is selected by this diagnostic for deployment.

For each candidate N, baseline B and labeled target Y, compare the binary oracle
min(L(B,Y), L(N,Y)) with min over 0 <= alpha <= 1 of L(B+alpha*(N-B),Y).
L is unchanged past-normalized twelve-waypoint mean Euclidean distance. Alpha
is one scalar for the full path, not one value per waypoint. Convex subgradient
bisection uses 36 fixed steps and a Lipschitz numerical loss envelope. This is
not interval-arithmetic certification or a statistical safety bound.

Also compute the per-row minimum over eight candidates within each fixed seed.
This is a union of line segments, not the full convex hull: arbitrary mixtures
of different candidate vectors can lie outside it and are not bounded here.
Any joint selector restricted to this action class cannot exceed the unconstrained
rowwise oracle's labeled-set gain. Pairwise penalties restrict available choices;
they do not manufacture missing prediction support. New forecasters may exceed
this action-class ceiling, and no impossibility claim about vision is justified.

Report three-seed/equal-physical-scene means and 2,000 descriptive scene resamples.
Report static-start, static-stay, moving-past and easy slices. Fixed relaxed
coverage fractions are 0, .01, .05, .10, .25, .50 and 1, using ceiling-rounded
per-scene row counts and top realized gains. These deliberately ignore local
query constraints and causal score uncertainty. They are not deployable curves.

All labels enter only the oracle evaluator. Private per-row alphas cannot be used
as model inputs or deployment thresholds. Original forecasts and reports are
unchanged. No development/calibration/confirmation data, new data admission,
Stage5C, SMC, metric/seconds claim or claim of learned oracle performance.

Synthetic checks cover endpoints, interior optima, nondifferentiable ties,
whole-path versus waypoint scaling, a dense-grid comparison, coverage aggregation
and a counterexample showing the bound does not cover arbitrary mixtures.
Recompute all real diagnostics exactly, and independently compare sampled real
paths with a separate scalar optimizer before relying on numerical accuracy.
