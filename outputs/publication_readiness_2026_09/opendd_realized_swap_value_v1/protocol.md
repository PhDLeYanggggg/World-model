# OpenDD: predict the outcome of the nominated swap

Use the same development locations, forecasts, exact within-query swap
nominations, and per-direction counts and movement range as the
[query mask study](../opendd_recipient_query_mask_v1/protocol.md).
Do not choose a different recipient. Compare the archived opportunity
mask, allocation on predicted signed trajectory gain, and allocation
adding half of the predicted signed relative-motion gain. Both gains
use native position units and average across the query and horizon.
The half-weight is fixed before scoring and is not fitted on held sites.

For each donor's legal swap, predict four targets: observed individual
benefit, individual harm plus missing-step debit, observed relative
benefit, and relative harm plus missing-pair debit. Use direct differences
against the incumbent. Each missing-step debit is the norm of the
forecast change at that step. Relative targets consider all affected
pairs, counting the swapped pair once. Normalize each target by its
available causal movement bound and restore the scale for ranking.

Train on all legal donor candidates rather than nominating on a donor
using a ranker trained on that donor. Each donor's proposals exclude
that donor and the held site. Only the frozen held-site nomination is
evaluated by the newly fitted heads. Use the same three seeds and 300
updates, equal donor draws, fixed checkpoints, and no held tuning.

Reproduce the archived opportunity result. Both learned allocations use
exactly 353 swaps and the original 95--100% movement range per direction.
Compare direct paired conservative and observed trajectory/relative
scores, strict gains, risk checks, and tails. Count qualifying swaps
separately from aggregate gain. Use 3,000 paired location resamples after
averaging seeds. This offline comparison preserves the risk rule, does
not qualify a release policy and does not use independent outcomes.
