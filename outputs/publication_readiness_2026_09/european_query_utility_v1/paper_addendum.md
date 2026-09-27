# Development Evidence: Allocation Is Useful but Risk Remains Unresolved

With predictors and risk estimators fixed, we compared independent admission,
uniform query admission, risk-unconstrained utility ranking and constrained
joint allocation. The main contrast fixes the number of interventions within
every current query, including rows without evaluable future labels. Joint
allocation maximizes predicted utility under two aggregate signed-excess
constraints for overall and easy-case risk. Failed optimization checks retain
the original independent action.

Across twelve already-opened development localities, joint allocation improves
ADE over independent selection by 0.2322% (exploratory locality-bootstrap 95% CI:
0.1225% to 0.3586%). All twelve locality point contrasts are positive, and three
forecasting-seed summaries agree in direction. However, 99 of 216 dependent views
violate the observed 2% positive-harm budget, compared with 82 for independent
selection; ten risk ratios remain undefined. Net easy error is preserved but
cannot serve as a substitute for positive-harm control. We therefore do not
deploy the joint policy or claim that its predicted constraints certify safety.

This development result motivates query-level risk modeling under source
shift. It is not independent confirmation, a physical-safety result, or evidence
that utility maximization itself is novel. The experiment changes utility
ordering and risk pooling together, and contains no pairwise conflict term.
Independent confirmation, identified and calibrated harm targets, complete
multimodal/interaction ablations and uniformly budgeted public method baselines
remain necessary for a main paper claim. Image-local detector silver and
raw-frame obs8/pred12 restrict the interpretation; no metric or seconds claim.
