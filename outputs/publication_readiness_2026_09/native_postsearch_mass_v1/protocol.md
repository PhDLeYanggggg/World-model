# Full-Recording Risk Mass After Selection

The earlier post-search probability head did not improve the
recording result. Use exactly the same 7/6/1 source roles, cross-fitted
search on the inner six of seven, and the same fixed search on validation
and held sources. Instead of classifying local budget events, fit
four nonnegative quantities: all/easy positive harm plus the original
unknown-outcome radius, and all/easy observed CV reference mass.

Compare the same ridge features and training choices under two losses:
mean normalized squared error per query, and the average of that
loss with exact full-recording normalized mass squared error.
The recording feature mean is weighted by the causal scale of each
selected assignment. This is not the older sampled-batch excess loss
on an amplitude bank: it uses full totals and the inner-out-of-source
choices made by exact joint search. This contrast is not claimed novel
on the basis of aggregation alone.

Both fitted heads use the same causal ledger. Start at zero expected
credit, process current frames in order, and emit a fixed search choice
only if both expected cumulative excesses remain nonpositive. A future
query may not pay for a current choice. Select the internal predicted
budget multiplier from 0.25, 0.5, 0.75 and 1 on the six validation sources:
maximize original passes, minimize total-risk violations plus undefined
easy reference, then prefer larger conservative OLS4 gain and fewer
switches. If all options emit nothing, retain fallback and report
support as undefined. Validation choice is exploratory unless every
required validation view passes. Scoring keeps the original 2% rule.

Primary: recording-mass policy vs. retrospective pointwise excess rank
at the same number of switches per recording and cardinality. Also
report the true causal pointwise ledger, the frozen matched search,
and paired recording-level normalized excess MSE on all nonempty
search emissions. Report the number of empty/unsupported views and
use 3,000 paired source bootstrap draws, averaging three forecaster
seeds. Same 34 exposed recordings, 14 sources, 102 dependent views,
8 observed/12 forecast annotation steps in image pixels. No external
input/outcome, new deployment, Stage5C or SMC.
