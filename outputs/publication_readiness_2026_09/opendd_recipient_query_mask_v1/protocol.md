# OpenDD development: find the query before ranking the swap

Use the same exposed OpenDD vehicle groups, frozen two-edit proposals,
native 8/12 steps and rotated one-recipient swap menu as the
[eligibility study](../opendd_recipient_eligibility_v1/protocol.md).
The query head uses the mean, spread and max of the causal descriptors
of all legal swaps, plus menu size. Its label is whether any swap both
improves conservative trajectory error over the incumbent and meets the
original risk and CV/OLS4 criteria. Only donor sites supply this label.
Each donor's forecast packet excludes that donor and the held site.
Fit on six donors with equal per-donor draws, three seeds, 300 fixed
steps and no held selection or recalibration.

Freeze a single within-query action for each legal query using the
archived v194 gain output trained without the held site. All compared
masks use these identical nominations, including on the archived mask.
Compare three masks: archived, top raw predicted gain, and predicted
query opportunity. Per held site, paired donor and seed, both new masks
choose exactly the archived swap count and spend 95--100% of the sum
of archived nominated swaps' peak displacements. This additional cap is
for reallocating swaps; every arm preserves the original stepwise
displacement and two edited forecasts in every query. Full causal score
lists and the archived count are available to the optimizer, so this is
an offline attribution, not a streaming policy or a release rule.

Measure query coverage, actual qualifying swaps, direct conservative
trajectory and relative-motion gains, full-roster original risk and
two-reference passes, worst observed easy degradation and available
error tails. Enumerate the budget-constrained hindsight optimum for
qualifying swaps using the same fixed nominations to distinguish missed
queries from poor within-query choices. Use 3,000 paired location
bootstrap draws after averaging three seeds. The original risk scorer,
external evaluation roles and deployment are unchanged.
