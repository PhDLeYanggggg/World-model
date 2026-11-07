# Shared Targets In The Pair Completion Bound

On exposed Moving A, replay the original joint, additive, independent,
shared-family, CV and OLS4 paths for three seeds. Preserve all requested
pair edges, co-observed labels, query weights, per-source weights and
3000 paired source draws after seed averaging. Verify the original
pointwise bounds before interpreting the new result. Group only within
each recording and seed by canonical agent pair and absolute annotation
step. Verify released label coordinates and masks agree for all repeated
recording-agent-step identities before grouping unknown pairs.

For a missing pair-step with shared target y, its weighted loss
contrast is the difference of two weighted sums of distances to y.
For any transport plan between the two sets of forecasts with the
original weights as its marginals, the absolute contrast is bounded
by that plan's Euclidean cost (reverse triangle inequality). The
diagonal plan is the old bound. For equal query weights use optimal
assignment; otherwise greedily exchange diagonal mass in symmetric
two-cycles only when this reduces cost. The latter is a feasible plan,
not an optimal-transport claim. Sum the resulting bounds across
targets. Original weights, including the number of requested edges
and steps in each query, are unchanged.

The primary contrast is joint vs additive. Also report joint vs
independent, shared family, CV and OLS4; keep both bounds and paired
intervals. The endpoint comparisons are unchanged because the fixed
lead does not repeat the same target across origins within a recording.
This is a development sensitivity on pair geometry, not a new method
for imputation, not a change to individual utility, and not a change
to the original recording safety rule or external admission criteria.
No new training, deployment, external role access, Stage5C or SMC.
