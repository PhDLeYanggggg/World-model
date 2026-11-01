# Decision-Cost Objective And Its Limits

Let a row have seven frozen forecasts and fallback a=0. The coupled network emits
B,H,R,R_easy,H_easy in past-normalized units. Only B,H,H_easy can change;R and
R_easy stay fixed. Its selected-query context and462 input fields are unchanged.
Define p=softmax(0,(B_k-H_k)/T),masking only causal candidate unavailability.
T is the fitting-only sqrt(RMS_B^2+RMS_H^2),not a test-selected temperature.

The old regression objective is0.5*(benefit_MSE+pointwise_risk_MSE+selected_query_score_MSE).
The new objective adds the expected actual intervention cost under p plus three
nonnegative per-training-record multipliers. Actual known gain is B_true-H_true.
Unknown-row gain is lower-bounded by negative causal forecast disagreement.
All/easy constraints use positive harm minus0.02*the selected reference,not net error.
Easy degradation is the selected easy net error minus0.02*the full known easy reference.
Unknown harms take the causal envelope and unknown references/benefits contribute0
in the conservative bound. This is not supervised zero imputation.

For source s with Q_s current-frame queries and known full reference R_s,the gain
row coefficient is Q_s/R_s. With equal-source/uniform-query draws,this recovers
the equal-source full utility expectation,even for unequal query sizes.
For record r within source s,the risk coefficient is Q_s/(N_records_s*S_r),where
S_r is the complete causal past-scale sum. This recovers equal-source/equal-record
constraint expectation. Tests enumerate all query draws against independent sums.
Every100 optimizer updates,exact whole-record inference recomputes all constraints
and updates lambda=clip(lambda+constraint/S_r,0,100). No sampled-risk sign is used
for that update. The easy reference constant affects the dual update but has zero
parameter derivative. The final2,000-update checkpoint is fixed before outcomes.

This is an empirical soft surrogate,not an exact differentiable MILP or a calibrated
upper confidence bound. Inference discards training duals and uses the original
hard physical eligibility and contemporaneous joint allocator. Calibration errors,
soft/hard disagreement and the frozen-context/final-selection gap can still fail.
Exact finite training-record expectations are not independent-scene guarantees.
Fallback-only records with undefined required selected risk do not pass by fiat.

Earlier binary cost-sensitive primal training already tested direct cost learning
in this repository. The new factor is its seven-action coupled-critic setting and
the matched calibration auxiliary;there is no claim to invent decision-focused
learning. [SPO](https://arxiv.org/abs/1710.08005) measures downstream decision error
and derives SPO+;we do not implement SPO+ or inherit its consistency result.
[Multi-expert regression deferral](https://proceedings.mlr.press/v235/mao24d.html)
provides bounded-cost surrogate theory,not the original native recording-risk
guarantee needed here. Bibliographic/abstract scope rechecked2026-11-01;the failed
arXiv HTMLv7 URL was corrected to the actualv5 abstract. No new full-proof audit.
