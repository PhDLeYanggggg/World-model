# Prepare the fixed-budget comparison before outcome access

Read only the verified Songdo histories and vehicle forecast bank.
Reuse the direct pair and additive heads trained on OpenDD before any
Songdo result was available. Average the 21 location-pair-excluded
heads within each of the same three training seeds. Average outputs
before applying the previously registered gain and risk ranking score.
This is a source-neutral set of proposed edits, not a calibrated policy.

Within each flight and seed, use the established OpenDD fixed-query
rule: select the query with the largest number of legal recipients
that supports two equal-radius edits and leaves at least one actor on
CV. Break ties by earliest input order. Keep the same query, same
paths, same radius and same two-edit count in all four arms:

* Independent: rank each action independently by its additive head
  output, then take the best two distinct actors.
* Additive: score both actions using the average of their additive
  logits and gains, then apply the shared rank score.
* Same family: use the joint head but require both edits to come
  from the same path family.
* Joint: use the joint head across all legal distinct-actor pairs.

The finite search enumerates unordered pairs and computes each actor
embedding once. All selected trajectories are stored as displacements
from their last observed coordinates. CV and OLS4 remain reference
forecasts; two edits are a matched comparison, not a decision to
release those edits. Flights without an eligible slot are recorded
explicitly. No future index, coordinates, loss, risk threshold or
role assignment. These proposals can be tested only after a
prospective external evaluation protocol is selected.
