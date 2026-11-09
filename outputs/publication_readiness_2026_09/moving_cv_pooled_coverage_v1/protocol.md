# Pooled Predicted Risk for Broader Forecast Replacement

Use the frozen fitting-source CV admission head, analytic training/calibration
paths, and held source-excluded neural donors. Expand the search from the
previous sparse shortlist to all legal row/action pairs in each current query.
No training, future inputs, or cross-query lookahead. Estimate each action's
two original relative risk excesses in common pixel units using the causal
scale. Prediction slack is not an observed risk guarantee.

The pooled controller first takes one action that predicts easy support and
nonpositive excess in both budgets. Then process remaining agents in descending
predicted gain order; take their best positive-gain action that fits the current
two predicted budgets, and debit or credit each budget. Keep at most one
action per agent. Return CV if there is no support donor or the total predicted
gain is nonpositive. Do not carry slack across queries. This is a greedy causal
policy, not an exact risk optimizer.

Held diagnostics use zero extra reserve, predicted easy probability at
least .10 for the donor, and caps eight, 32, and no artificial cap. Compare
with frozen sparse choices, independent full-menu screening at the original
fixed relaxed screen and at zero predicted excess, unconstrained gain, and a
gain-only control matching the uncapped pooled count in every query. The
control matches counts, not displacement radii. Evaluate original recording
risk and conservative CV/OLS4 and pair utility, with 3,000 source-bootstrap
draws after averaging three seeds.

The calibration-only grid combines caps eight/32/no cap with extra normalized
excess reserve 0/.005/.02 and donor minimum predicted easy probability
.10/.25. Require all calibration recordings to pass the original risk rule
and positive source-macro lower gain against both references. Otherwise the
policy returns CV without certifying support. Freeze choices for all nine
cells before held scoring. Keep A outcomes, DroneCrowd independent
confirmation, the original 2% rule, and the 15/30 split unchanged.
