# Same-Query Allocation: Original Run and Numerical Repair

The full registered V1 run and its exact replay are complete. All 175,756
past-eligible windows, three seeds and three frozen predictors were included;
188,388 query/action/seed instances are not independent scenes. Separate
raw-label arithmetic and 1,190 small-query optimality checks pass.

V1 contains 127 query instances with failed numerical solver postchecks. Their
proposed solutions were rejected and replaced by CV, not admitted as safe. This
also invalidates exact-count joint/unary comparisons in those queries. All V1
choices, metrics and failures remain unchanged in this directory.

[Solver diagnosis](solver_diagnosis.json) and [primal inspection](primal_diagnosis.json)
reproduce 14 fixed causal cases: all 14 rejected calls had small risk-row overruns,
with no integer or product mismatch. The subsequent
[versioned numerical repair](../easy_allocation_risk_scaled_v1/conclusions.md)
changes cost units equivalently, retains the original risk check and repairs only
the 127 affected queries. Use that repaired readout for matched-count scientific
interpretation; do not silently overwrite or pool versions.

Neither version beats the old strict neural policies or establishes meaningful
nonadditive-agent prediction lift. Fresh allocation and evaluation use
cached-verified forecasts and heads; no new model training, threshold selection,
external readout, calibration, confirmation or deployment occurred. Four exposed
SDD sites, obs8/pred12 annotation pixels, not historical t+50, seconds, metric,
true 3D or foundation-model evidence. Stage5C and SMC remain off.
