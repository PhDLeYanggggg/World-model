# Decompose the already scored Songdo contrast

Use only the 160 confirmation flights and three seeds in the original
completed run, with the exact pre-outcome forecasts and release freeze.
This is a retrospective explanation, not an additional test or an
opportunity to refit. Do not open the calibration flights, other
external cohorts, or change any recipient, score, threshold or rule.

For joint minus each of independent, additive and same-family, split
the edited query into joint-only recipients, control-only recipients
and shared recipients. Observed errors and worst-case missing-waypoint
debits both add across the three sets. Reuse the original whole-flight
CV denominator and query weights. Reproduce the primary score for
every flight and seed before aggregating. Average seeds within flight,
flights within linked source component, then use the original 3,000
source bootstrap draws. This separates which additive contribution
is visible under the frozen choices, not the causal effect of forcing
the same recipients on both models.
