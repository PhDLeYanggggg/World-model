# Fixed-Batch Scope Clarification

The protocol's phrase "saved final fixed fitting batch" refers to evaluating
the saved `fixed_ids` at the final checkpoint. These256 draws were established
at initialization and reused for diagnostic loss traces. They are NOT the
last stochastic optimizer minibatch. The runner already uses those exact
saved IDs, and tests/receipts bind them; no sampling or computation changes.

The gradient calculation therefore describes one fixed fitting diagnostic
batch under final parameters. It cannot identify the full training trajectory,
gradient-noise distribution, causal influence or optimizer response. This
wording clarification changes no data roles, thresholds, outcomes or gates.
