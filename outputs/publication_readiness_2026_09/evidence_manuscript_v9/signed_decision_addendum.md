# Signed-Cost Admission Objective: Negative Development Evidence

We compared squared-error continuation with a magnitude-weighted logistic
surrogate for the sign of all/easy budget excess. The six paired neural heads
share frozen kinematic initialization, preprocessing, queries, parameter count
and 2,000-step budget. Unknown-outcome bounds retain the original MSE. A full
treatment refit and prediction/action/metric replay are exact; no new trajectory
predictor is trained. Models are frozen before exposed SOURCE validation.

The surrogate reduces fitting easy-channel cost-weighted sign regret by
81.03-85.10% relative to MSE. This does not translate into safe held utility:
equal-source known ADE gain over the physical floor is 2.6928% versus 3.6006%,
hard gain is 3.5649% versus 4.6257%, and worst defined easy-net degradation is
26.2254% versus 38.9046%. All-risk violations decrease (14 to 11), but easy-risk
violations increase (13 to 14). Full risk/support passes are 1/17 versus 0/17;
the single passing treatment view has only 13 selected rows out of 3,193.

The paired known contrast is -0.9078 pp, bounded by [-2.2534, +0.4379] pp under
missing-label completions. These are completion bounds, not sampling confidence
intervals. Seven exposed SOURCE units and one producer seed constitute exploratory
development evidence only. The experiment motivates checking feature support
and boundary transport, not a claim of safe generalization or method novelty.

The original 2% and support rules are unchanged; original24group primary remains
NA, empty matched utility controls remain NA, and no identity certificate is used.
No reserved H/domain evaluation, deployment upgrade or submission-ready claim.
Native obs8/pred12 image pixels; no verified metric/seconds, true3D or foundation
claim. Stage5C and SMC remain disabled. Complete evidence is in
`../moving_signed_decision_v1/`; full historical tests were not rerun.
