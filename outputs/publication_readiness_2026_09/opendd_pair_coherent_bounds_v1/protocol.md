# Check whether a shared missing position tightens relative-motion bounds

This is a post-score OpenDD development audit, not a change to the
release protocol. Reuse the two fixed recording masks and three
comparisons in the completed pair-angle bounds audit, with no new
fitting or choices. Reproduce its independent-pair endpoints.

At each time, group edges from a missing actor to observed actors.
They share one unknown position. The old bound pairs reference and
prediction distance anchors by the observed actor identity. Instead
find a minimum-total-distance bijection between the two sets of
anchors. The summed distances are invariant to reordering. Keep the
previous debit for edges whose two actors are missing. Do not assume
track smoothness, a motion model, or missingness at random. This is
an outer interval, not necessarily the set of all achievable values.

Separately compute two feasible values per changed query by testing
actual coordinate completions with observed positions fixed. Opposite
signs witness ambiguity only within that query. Do not combine those
completions across seeds or claim a common completion of a recording.
Report changed views, contraction, newly resolved directions and
seven-location paired bootstrap intervals at both quotas. Do not open
pNEUMA or other new external outcomes. Keep the original risk and
reported policy decisions unchanged.
