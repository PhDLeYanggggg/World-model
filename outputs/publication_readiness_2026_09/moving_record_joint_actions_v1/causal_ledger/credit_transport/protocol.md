# Does Estimated Credit Transfer Across Queries?

Keep the chronological full-bank ledger, source-excluded ridge costs,
original pointwise query counts, eight alternatives, and original 2% risk
readout. Change only how much **negative query excess** is carried forward.
Same-query netting still uses the unmodified estimated costs. Positive
costs are debited in full; later queries do not affect earlier choices.

Four matched arms: no carry, full carry, fixed 50% carry, and separate
source-calibrated all/easy carry. For each held source and seed, apply the
full matched ledger to each of the other 13 donors using that donor's
leave-two-sources-out cost model. Neither donor nor held source contributes
to the inner ridge fit. On the queries that *predict* a credit for one
budget, sum signed actual completion savings and divide by predicted
savings per donor source. A positive realized excess offsets real savings;
zero selected easy reference is not imputed. Clip each ratio to [0,1].
Transport the lower empirical 10th percentile across those donor sources
with credit support, or zero if fewer than three. This is an empirical
discount, not a coverage or safety guarantee. No threshold grid or model
choice is made using held outcomes.

Reproduce both endpoint controls (no carry and full carry) exactly. Freeze
the four arms before scoring. Primary contrast: donor-discount versus full
carry at the identical query counts. Also report fixed half carry, direct
paired completion utility, every original risk/support view, and the
diagnostic of realized excess on frozen predicted-credit and spending
queries. The latter reads labels only after freeze; it never updates the
ledger. Three actual seed banks, 14 exposed-A source groups, 3,000 paired
source bootstrap draws. No external outcome, original rule change,
deployment, Stage5C or SMC.
