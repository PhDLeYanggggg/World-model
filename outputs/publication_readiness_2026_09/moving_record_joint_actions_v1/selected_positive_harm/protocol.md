# Selected Positive Harm: Matched Direct And Factorized Learning

The previous nested calibration identified easy positive harm, not a
uniformly inflated reference, as the dominant selected-cost error on A.
Instead of tightening a threshold, replace just the known easy positive
harm estimate. Unlike earlier easy-membership/reference factorization,
supervise the event that a known easy row incurs positive harm, and the
harm magnitude conditional on that event. Preserve the old selected
reference, unknown contribution, overall excess and utility estimates.

For each held A source and seed, use the other 13 sources for fitting.
Form each donor's admissible pool with its own nested leave-two-sources-out
cost model, excluding both the held and donor source. Within that causal
pool, train on known easy positive harm divided by causal path radius.
Wholly unknown rows have zero known-harm contribution, not an observed
zero-harm label; their separate unknown-cost estimate is not replaced.
Inputs are the identical causal descriptors, action ID and frozen cost
features, standardized using the fitting pool only.

Two matched 32-wide Torch heads use the same initialization, donor/query/
action draws, AdamW, 400 updates and fixed final checkpoint for every
source and actual producer seed. Both predict a product of two sigmoid
outputs and use fractional cross-entropy on that expected harm. Only the
factorized head also supervises event occurrence and conditional severity
with auxiliary weight .5. Three seeds, 84 fits, 33,600 updates. Do not
select a model or loss weight on held A outcomes.

Freeze the original admissible pool and exactly its selected count per
query. Four arms: raw filter-first ranking; ranking using the same utility
minus easy excess with the frozen ridge harm; with the direct harm head;
and with the factorized harm head. At the matched counts this is a ranking
test, not a new safety certificate or free-count policy. The only change
between the last two arms is the harm loss supervision.

Primary: factorized vs direct paired completion lower at matched counts.
Mechanism: known easy positive harm at the same counts, normalized by the
full known CV mass (not the original risk denominator). Also measure
harm-fraction MSE on fixed raw selected actions, raw matched scores, all
102 original recording risk views and 3,000 source bootstrap draws after
seed averaging. Entire A roster, native 8/12 image coordinates. No external
outcome, change to the 2% rule, deployment, Stage5C or SMC.
