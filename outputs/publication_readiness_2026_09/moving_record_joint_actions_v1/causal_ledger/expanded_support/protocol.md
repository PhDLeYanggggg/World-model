# Expand Decision Support Without Relaxing The Observed Risk Rule

Reuse the frozen Moving A roster, 8 observed and 12 future annotation
steps, all eight original paths, three actual forecast-bank seeds and
existing source-excluded cost estimates. Do not retrain or select any
threshold. Every compared policy keeps the same original switch count
at every query. The original 2% observed rule, support, missing-outcome
radius and signed easy-net check are unchanged.

First compare the same fitted constrained-teacher membership ranker
under the previous spend-dominant shortlist and the full candidate
bank with its chronological predicted allowance. The latter may use
earlier predicted slack but not later frames. Each query has its
frozen zero-credit count witness. Replay both old decisions exactly
before interpreting the comparison. No trajectory or ranker refit.

Second compare the original full-bank utility objective to two
prespecified objectives, keeping its predicted risk costs and
chronological optimizer fixed: utility plus 0.25 times predicted
selected easy reference; utility minus 0.25 times predicted easy
excess. Each term is divided by the same causal motion scale.
The latter is a generic caution control. These are intervention
ranking bonuses, not changes to the risk limit or calibrated claims.

Primary model-side readouts: the paired conservative utility
contrasts, original recording passes, and defined observed easy
reference mass; source bootstrap 3,000 draws after averaging
forecaster seeds. All 102 repeated recording views are kept.

Separate structural diagnostic: hold the original ledger's past
predicted slack fixed at each query, then allow every candidate set
under that current allowance. Minimize the actual all/easy budget
excess separately and maximize actual easy reference. Sum across
the recording. A positive sum of minima or no possible support
proves impossibility in this fixed-allowance relaxation. Not ruled out
does not mean feasible: one set may not optimize both budgets,
and choosing it could change a later prefix. This is not an outcome-
aware policy and it exports no choices. Compare the narrow bound
without changing the original protocol. No independent outcome
access, deployment, Stage5C or SMC.
