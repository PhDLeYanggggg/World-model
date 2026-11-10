# Fixed-count eligibility for OpenDD recipient swaps

Use the exact same one-swap menu, native 8/12 observation/prediction,
displacement at each prediction step, and exposed seven OpenDD locations
as the [recipient transport study](../opendd_recipient_transport_v1/protocol.md).
No new external outcomes. The comparison uses the 353 recorded instances
where the archived mean-loss head swapped one recipient. The new rankers
can change which swap is made on these exact instances, not whether or
where to swap. An optional guarded version can skip a swap and is reported
separately, never as a matched-rate improvement.

Fit a shared two-output network with a Huber gain loss and binary
cross-entropy eligibility loss. Gain is direct conservative trajectory
improvement over the frozen additive incumbent divided by the swap's
summed movement. Eligibility means the original recording-level 2% risk
and support requirements pass and the candidate beats both CV and OLS4.
Compute both labels only on donor sites. Never pass risk denominators,
future positions or validity into the model. The frozen proposal producing
each donor's inputs excluded that donor and the held site. Preserve three
seeds, equal donor draws, 300 fixed updates and checkpoint every 50.

At equal swap count, compare the archived mean-loss head, the new head
ranking only by predicted raw gain (normalized gain times movement), and
the same new head ranking first among options with predicted eligibility
at least 0.5. The latter chooses the highest predicted eligibility if
none meets 0.5, so the fixed count remains exact. A separate guarded arm
keeps the incumbent unless eligibility is at least 0.5 and predicted
gain is positive. These are fixed decision rules, not fitted cutoffs or
safety guarantees. No held outcome selects a threshold or checkpoint.

The primary attribution compares risk-aware versus gain-only at the
same swap mask: strict safe improvements, original risk, both reference
gains, direct conservative trajectory gain and relative motion. Report
the available hindsight opportunities within the frozen swap mask, and
compare the guarded arm and both matched arms with the frozen additive
incumbent. Use three seeds, then paired 3,000-draw location intervals.
The same recording appears in multiple directions and seeds. Preserve
undefined support and use the original risk evaluator; do not change
deployment or open external datasets based on this development readout.
