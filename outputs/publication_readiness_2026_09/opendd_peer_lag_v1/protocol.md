# Prefix-validated lagged neighbor motion

The last experiment compared own and neighbor velocity transients at
the same edit radius and found no reliable difference. A neighbor may
provide information only when its past motion predicts the target's
next change. Here the support check takes place wholly inside the
eight observed steps. It does not use a forecast label.

1. Choose the nearest co-directional neighbor using only the first six
   positions. A legal neighbor has displacement cosine at least 0.8,
   separation no larger than the mean of the two prefix displacements,
   and is in the same query with at least three agents.
2. Predict each change in velocity from either the previous change of
   the same agent or the previous change of the neighbor. Each arm
   uses one scalar coefficient, fitted on the first three lag pairs
   with the same nonnegative capped ridge formula. Require the peer
   model to beat both self lag and zero on the final two already
   observed positions. Use exactly this support for both arms.
3. Refit the scalar on the five observed lag pairs. Add its predicted
   next velocity change as a decaying acceleration impulse on top of
   the same own-transient trajectory. Use a two-step half-life for the
   acceleration impulse and four for the base velocity transient, so
   matching the radius need not erase a change along a straight path.
   Both paths get the smaller of
   their two natural peak displacements relative to CV. A path may
   fill a shorter slot, never a larger one.

Freeze these candidates before reading forecast labels. Reuse the seven
OpenDD development locations, three seed budgets, CV/OLS4, common menu,
singleton decisions, matched additive reference and original risk
checks. Compare the constrained relative-motion ceiling after adding
one candidate from either arm. Then substitute each candidate at the
same original additive recipients and radii on two-edit queries. Record
whether the final two observed steps actually create a usable slot,
which matters independently of performance on the forecast labels.

Report full-grid and observed-only results as before. Use 3,000 paired
location bootstrap draws for direct conservative accuracy and full-grid
pair gain. No threshold selection, newly trained neural checkpoint,
confirmation split, exposure to reserved outcomes, or deployment change.
The lag is a predictive ordering, not evidence that one agent caused
the other to move.
