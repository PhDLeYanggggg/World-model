# Additional Aggregation Limitation, Identified During Training

The registered scaling experiment remains unchanged. This note records a
separate structural limitation; it does not select a new policy or modify the
ongoing objective after observing results.

The TRAIN risk surrogate averages agent excess within each query, then queries
equally. The development completion-risk function instead sums selected agent
harm and reference mass. With variable agents per query, these constraints are
not equivalent. Positive scaling preserves the TRAIN surrogate zero set, not
equivalence to the different agent-mass readout constraint.

A unit counterexample uses two fully known easy queries: the singleton has
reference 100 and harm 1; the other query has ten agents each with reference 1
and harm 0.22. Query-mean excess is (-1+0.2)/2=-0.4, which passes. Agent-mass
excess is -1+10*0.2=1, and selected harm/reference is 3.2/110=2.909%, which fails.
The test exercises the actual training reduction and completion-bounds code.
This is synthetic mathematical evidence, not an observed dataset result.

Consequently, even perfect TRAIN query-surrogate feasibility cannot certify the
readout risk, apart from the already known soft/hard, missing-label and domain
differences. This is a confirmed mismatch of aggregation definitions, not proof
of how much of the empirical failure it explains. The current single-factor
run must finish unchanged. A subsequent experiment should quantify the mismatch
on fixed TRAIN queries and test a readout-aligned risk estimator without changing
the official risk budget, missing-label handling or independent evaluation roles.
