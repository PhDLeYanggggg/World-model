# Fixed Numerical Repair of Easy-Risk Allocation

This is a development-exposed numerical repair, not a new threshold search or an
independent replication. The V1 allocation, results and negative comparisons are
retained unchanged. Diagnosis used causal solver inputs only. All 14 sampled
rejected calls violated the original risk row by 5.6e-8 to 8.2e-7; integer and
product errors were zero. The original postcheck correctly failed closed.

Repair only the 127 query/action/seed instances flagged unmatched in the frozen V1
decision manifest. Keep all other decisions byte-equivalent. Recompute all three
population, unary and joint choices within affected queries, so their exact count
remains paired. Multiply gains, harms, geometry weight and harm budget by the same
positive numerical factor, selected only from the causal risk row. This is an
equivalent cost-unit change, not a change in support, objectives, geometry tradeoff,
rho=0.02, denominator scope or feasible decisions. Retain primal/dual certificates
and verify risk and objective in original units. Never loosen the risk postcheck.

Predictors, forests, source exclusions, seeds, queries, labels, bootstrap unit and
analysis remain frozen. There is no refit, model selection, outcome-dependent case
selection, calibration, external readout, confirmation access or deployment.
All decisions must complete before recomputing outcome metrics. Replay all repaired
decisions and independently recount budgets and raw-label metrics. Retain failures
if any remain; do not call unmatched queries matched interaction evidence.

SDD annotation pixels; observed 8 / predicted 12 native annotation steps, stride
12. No seconds, metric, 3D, foundation or physical-safety claim. Stage5C and SMC off.
