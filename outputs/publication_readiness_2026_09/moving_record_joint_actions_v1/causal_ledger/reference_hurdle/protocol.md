# Support and Value for a Single Causal Insertion

The earlier learned shortlist improved hindsight capacity, but no
threshold recovered observed support. Test whether explicitly separating
reference availability from gain is more useful than ranking them jointly.
This is a development experiment on the already-exposed own-B cohort,
not an independent test.

Reuse the same frozen forecasters, direct policy, seven fitting sources,
six calibration sources and one source-held evaluation per fold. Train
on the same candidate actions: unselected, nonidentical forecasts in a
query with space under the original cap. Features use past motion,
frozen rollout descriptors and predictions from fitting-only upstream
cost heads. Cross-fit those costs within the fitting sources. No
outcomes or recording totals enter the candidate features.

One 32-hidden-unit network per fold, 220 updates at batch 256, outputs
observed easy-reference probability, signed value conditional on easy
reference, positive harm conditional on no easy reference, and direct
value. The scalar value target is easy gain minus non-easy positive
harm, with per-action normalized magnitudes clipped at 3 for fitting.
For wholly unknown labels, the original displacement disagreement is
charged as harm. Compare support-only ranking, direct scalar value,
and probability-weighted conditional value minus non-easy harm. The
network, features, candidate roster and fitting draws are shared.

Each arm ranks one candidate per query and, if allowed, inserts the
first candidate meeting its support and value cutoffs. No deletion,
future-query access, or replacement of an existing action is allowed.
Choose cutoffs only on calibration sources: require additional
recording passes, no lost passes, no new signed easy violations,
and nonnegative paired lower utility. Use no insertion when no
cutoff qualifies. Evaluate the frozen choices with the original
recording scorer and unchanged 2% and support requirements.

Report observed easy support and nonnegative supported gain at the
first eligible query as separate unthresholded probes. The primary
component contrast is hurdle versus direct value for nonnegative
supported gain at that query, with a paired 3,000-draw source
bootstrap. Report whole-recording passes, edits, unsupported
views, and paired lower utility separately. Three forecast seeds
repeat the recordings; they are not independent observations.
Do not upgrade deployment or change the contract on this cohort.
No metric, seconds-level, Stage5C or SMC claim follows.
