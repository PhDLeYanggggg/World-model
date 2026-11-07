# Distill Donor Choices Under The Actual Risk Budget

Fixed exposed Moving A population, eight past and twelve future native
annotation steps, image coordinates, frozen full eight-action bank,
seeds 17/29/43 and original chronological inference allocator. Use
true outcomes only to construct training examples from donor sources.
No donor selection is generated with a model fitted to the held source.

For each donor current query, solve the zero-credit 0/1/2-switch problem
exactly using observed conservative utility and the original all/easy
2% positive-harm excesses, with the existing missing-outcome radius.
Do not require positive easy reference support in each query: the
original requirement is still measured on the entire recording.
An empty teacher provides no ranking label and is counted, not certified.
A query with no legal negative also provides no ranking label.
The matched gain-only teacher solves the same query at the exact same
count, maximizing conservative utility but ignoring harm constraints.
It is a training control, not a valid deployment rule.

Both students use the same causal descriptors and centered ridge
membership loss. At each nonempty teacher query, selected legal
actions share a fixed total target mass (mean one over legal actions),
so large queries do not dilute their positive supervision. Ridge 0.002;
equal active-query and donor-source weights. Each prediction excludes
all teacher labels and fitted moments from the evaluated source.
Keep the original predicted risk costs, chronological accounting and
per-query switch counts exactly matched in all evaluated arms.

Primary: constrained-demonstration student vs matched gain-only
demonstration student. Secondary: each vs original frozen ledger,
change in all/easy positive-harm completion upper, signed easy net,
hard utility, unknown selections and pair consistency. 3,000 paired
source bootstrap draws after seed averaging, all 102 recording views.
These are relative-to-baseline errors, not physical safety. A training
demonstration is not model performance or evidence of risk calibration.
No change to thresholds, independent-data roles, Stage5C or SMC.
