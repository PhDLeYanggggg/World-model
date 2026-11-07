# Missing-Position Acquisition Budget For The Fixed Pair Contrast

Use the existing 34 exposed recordings and three frozen seeds, same
joint and additive forecasts, edge lists, query weights and equal
source averaging. Read the released availability masks, not target
coordinates. Identify a position once by recording, agent and absolute
annotation step. Reproduce the original pointwise disagreement radius.

For each affected missing pair-step, its original uncertainty mass is
the Euclidean distance between the two forecast pair vectors times its
unchanged evaluation weight. Both missing endpoints must be resolved
before a term is counted. Compare chronology (recording ID, frame,
agent), descending incident mass, and a greedy allocator that chooses
the singleton or linked pair with the most newly resolved mass per
requested position. Prefix curves are evaluated at the same budget.
For each of 50, 75, 90, 95 and 99 percent removal, solve a continuous
minimum-budget relaxation with term indicator no greater than either
required endpoint indicator. A relaxed solution is a lower budget
bound, not an integer witness.

These are counts of positions, not human labor or tracking episodes.
They are conditional on each proposed position being independently
annotatable. Actual labels could shift the observed effect in either
direction. These calculations remove mass from the original pointwise
debit only; they do not claim a given reduction of the already tighter
shared-target bound. No annotation is collected, no policy or scientific
metric is changed, and no external role is read.
