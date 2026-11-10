# WEPDTOF: why the frozen pair penalty did not change selection

This is a forecast-only diagnostic. Reconstruct the original
fixed-count legal menu and frozen donor-head scores from the
registered camera prefixes and forecast bank. Do not load any
outcome, fit a model, or choose another coefficient. Confirm the
reconstructed selections reproduce the frozen forecasts exactly.

For each query use the original utility (U): the minimum summed
predicted gain across the two references, minus positive summed
risk excess. Let (P) be the original proximity-weighted mean
relative departure from CV. WEPDTOF's comparison is (U-0.10P)
against (U). It removes the entire pair penalty, not just the
cross term of the separate squared-relative formulation.

For the unpenalized winner (a), another choice (b) with
(P_b<P_a) can strictly beat it only above
((U_a-U_b)/(P_a-P_b)). Take the minimum such ratio. An empty set
means the winner already minimizes the penalty. Report one-edit
queries separately, and preserve the original stable tie rule at
the registered coefficient. A crossover is a property of the
frozen scores, not evidence of a better forecast or a recommendation
to increase the weight. No risk, role, or release rule is changed.
