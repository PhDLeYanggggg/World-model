# Fixed Native-Forecast Joint Mechanism Comparison

This source-only development registration precedes decision generation and
readout. Four admitted physical sites are already design-exposed, so it is not
an independent test. All frozen forecasts, source roles, native ADE/FDE metrics
and exact-zero-CV empirical protection criterion remain unchanged. No new model
is trained, no threshold selected and no deployment promoted in this study.

## Fixed Hypothesis and Controls

Test whether non-additive pair coupling changes useful forecast decisions
beyond geometry-aware independent selection at the same cardinality and
predicted-harm budget. Use all175,756queries,33recordings,20,932scene queries
and seeds17/29/43. Original val/test/main/external/bookstore roles stay closed.

Every candidate is the registered native-loss neural forecaster; the reference
is causal CV. The eligible proposal pool is exactly the frozen MSE strict rule
with the past-stop veto. Candidates outside that pool cannot be introduced by
the optimizer. Keep original uncontrolled-neural and all-CV controls.

For each scene query, full count equals the eligible pool size. This has one
feasible assignment and is a structural null control. Half count is floor of
half the pool size. Its reference selects greatest predicted net gain, with
source-agent-ID ties. This does not claim half coverage is optimal. The harm
budget is that reference's summed coherent predicted harm divided by total
visible context count; it is a matched input-only anchor, not a calibrated risk
tolerance. All optimizers share the exact count, support and this budget.

Compare risk-only independent, unary-geometry-independent and full joint controls.
The existing decomposition removes only binary products; signed product terms
are not clipped. At count0/1 or absent supported products, joint equals unary
analytically. At full count the fixed proposal set is the only feasible choice.
These no-op cases cannot support an interaction claim. Scene-uniform means all
forecast targets or CV under the same eligibility, count cap, budget and a
negative constructed objective; it is not secretly a subset proposal.

## Past-Only Geometry and Unknowns

Use the verified identity-resolved native_scene_context_v2 cache. Target CV and
neural forecasts are restored with the historical stored metric scale. Context-
only agents retain supported past-derived CV, never an invented neural forecast.
Where two past observations are absent, forecast support stays false. No future
target or future-validity mask determines proposals, graph edges or counts.

Reuse the engineering geometry convention: graph radius is the median past
scale of forecast targets; proximity threshold is0.1times that radius. The pair
proxy is mean squared positive proximity deficit, clipped to nonnegative excess
over the all-CV pair. Pair weight is1 after dividing native gain/harm scores by
the frozen head's complement-training cost scale. This nonphysical diagnostic
weight is fixed, not chosen from held outcomes. Unknown-forecast edges are
unpriced and counted, never declared collision-free. Proxy decrease is not
forecast improvement, realized-risk control or physical safety.

## Evaluation and Failure Rules

Generate and checkpoint every decision before loading future target arrays for
this readout. Reproduce every frozen full-count choice hash. Use the same label
masks for all arms, retaining partial and absent outcomes. Primary contrast is
half-joint minus half-unary in equal-scene native ADE gain against CV, with3000
paired physical-scene bootstrap draws after averaging three seed errors. Report
per-seed/site effects, FDE, positive-easy/hard diagnostics, exact-zero absolute
harm, tail errors, intervention counts, selected unknowns and known-context
proxy. Four explored scenes cannot establish generalization or risk calibration.

Numerically checked solver failure returns CV and is explicitly unmatched; do
not exclude the resulting rows or silently lower the requested count. Report
primal/dual checks, product opportunities, changed identities, identical controls
and zero-count scenes. Fixed-count comparisons are diagnostics, not deployment.
No favorable-seed selection, threshold search, tolerance relaxation or new test
access follows a negative result. All predictions remain8observed/12predicted
sampled annotation steps in SDD pixels. No metric/seconds/true3D/foundation claim.
Stage5C and SMC remain off.
