# Learning Which Edits Preserve Easy Reference Support

Development experiment on the same seven OpenDD locations and the same
frozen self-velocity forecasts and uniform coefficients. Reuse the
21 risk heads and pair-excluded donor checkpoints from the prefix study.
Fit 21 new Extra Trees probability heads (32 trees, depth 7, leaf 12,
single thread), using identical site-balanced donor weights and causal
features. The label is whether this fixed edit is known easy and its
positive observed harm is <=2% of its own CV error. Train only on
nonidentity donor edits, with each donor's train-only easy cut. Neither
availability nor affordability labels are passed to inference.

The reference policy is the previous query-local risk rule. Until a
recording has accumulated one expected affordable easy edit, rerank
its current-query choices by summed predicted event probability, retaining
the original predicted all/easy risk constraints and exact original edit
count. Ties use predicted gain. After reaching one, use the reference
policy. A matched heuristic replaces the event probability with predicted
easy reference mass divided by predicted all reference mass. These are
bookkeeping scores, not guarantees that a recording has a qualifying edit.

An additional support-first arm can add one predicted-risk-feasible edit
when the reference would abstain and a candidate's predicted event
probability is at least .25, while the running support score is below one.
It uses the same forecast and moment limits but is not count-matched.
No threshold search or outcome-dependent runtime choices. The state is
updated using predictions from already processed queries only.

Report event AUROC/AP, prevalence and Brier score on eligible held-site
targets, exact per-query count comparisons, observed ADE/FDE, paired
full-grid lower gain and site bootstrap (3,000 resamples, three seeds).
Check the original 2% rule on all 606 flight-seed views, including
undefined selected easy reference. Compare paired passes and regressions
relative to the original local rule. An empty matched utility view stays
undefined. No independent confirmation, deployment, Stage5C or SMC.
