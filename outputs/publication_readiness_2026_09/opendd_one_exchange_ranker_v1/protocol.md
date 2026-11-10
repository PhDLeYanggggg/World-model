# OpenDD: score a single exchange without reading outcomes at selection

Keep the frozen OpenDD nested menu, recorded edit count and movement cap
for every view, the chronological credit rule, the original 2% scorer,
seven location holds and three seeds. The comparator is the label-blind
offline matched allocation. Each tested plan either leaves it unchanged
or removes one edited path and adds the next legal path at another query.

Train on the six donor sites only. Use their archived ranker fitted
without the donor or held site, with the matching five-site easy cutoff.
Generate donor anchors with the two fixed label-blind token schedules
(greedy one-edit and seeded random priorities). Enumerate feasible
exchanges and uniformly sample up to 32 per anchor before seeing their
scores. The original trajectory scorer labels rescue, tie or destruction
relative to the donor anchor. Use transition weights 2, 1, 2 and an auxiliary
paired conservative gain target clipped to +/-1 percentage point for fitting
only. Fit for 400 updates with no held threshold search. Compare two
heads trained on the same examples: the two affected queries only,
versus also a pooled summary of the anchor edits and available queries.

The primary head selects the highest rescue-minus-destruction score
among candidates with positive predicted gain, if its score exceeds
the fixed margin 0.10. Also report the same rule without the gain guard,
the forced best candidate, and the outcome-blind smallest-movement
exchange (only if it reduces movement). Keep the no-exchange arm. All
features are observed histories and the frozen forecasts; the whole
history-derived proposal sequence and original spend are available, so
this is not a streaming policy. Evaluate every view with the original
scorer. Report added and destroyed dual passes, risk passes, undefined
risk, paired conservative gain, and 3,000 paired location bootstrap draws.
No external labels, release decision, or evaluation rule change.
