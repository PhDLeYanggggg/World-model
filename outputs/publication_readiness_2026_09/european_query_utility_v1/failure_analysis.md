# Allocation Failure Taxonomy

| Hypothesis | Observed evidence | Interpretation and boundary |
|---|---|---|
| Utility magnitude was discarded by sign-only admission | Same-count joint versus independent ADE gain 0.2322%, positive locality CI | Allocation repair has a small positive development effect; pooling and ordering are not separately isolated |
| Lack of solver feasibility caused the failed risk gate | 181,683 MILP query views, 22 checked fallbacks; accepted allocations obey predicted constraints | Numerical fallback was preserved; predicted feasibility alone cannot fix score misspecification |
| More interventions explain the gain | Exactly equal counts in each current query for independent/joint/top-k/matched-risk arms | Count increase cannot explain this paired allocation gain |
| Pooling predicted slack guarantees real harm control | 99/216 views exceed 2%; 10 ratios undefined | Rejected; learned signed scores are not valid source-transfer risk bounds |
| Better net easy error means positive-harm safety | Worst joint easy gain +0.3716% versus CV, yet risk gate fails | The two quantities are different; favorable outcomes must not cancel positive harm in the risk account |
| Utility-only selection is a safe stronger baseline | Top-k ADE gain 1.6807%, worst easy -3.2797%, 186 risk violations | Unsafe diagnostic only; never selected for deployment after readout |
| Very sparse whole-query admission solves the problem | Coverage 1.1243%, 59 violations, 46 undefined views | Lower coverage is not a safety certificate; uniform arm is not count matched |
| Incomplete labels are harmless | 28.0787 unknown joint interventions per dependent view | Their action counts are known, outcome risk is not; no zero-error imputation |
| Twelve positive locality means establish independent generalization | Twelve already-opened development localities; overlapping producer contexts | Encouraging development consistency, not independent confirmation or a multiple-comparison-adjusted result |

The prediction target is signed harm excess at the 2% budget. The four output
components are nonnegative score bases, not separately identified expected
cost moments. The joint solver sums these learned excesses. The current
experiment demonstrates that a nonpositive predicted sum can coexist with
positive observed excess on new source views. It does not identify whether
the primary error is aggregation mismatch, source shift, score bias, missing
labels or sparse-denominator uncertainty. Those need a source-separated
query-risk repair, not outcome-selected thresholds.

The retained query roster may omit other visible agents. No pairwise conflict
term was introduced, so this experiment says nothing about collision safety
or a learned nonadditive agent interaction mechanism. Raw-frame detector
trajectories are not metric or temporally calibrated physical trajectories.

Independent selection/calibration/confirmation are closed. No deployment,
Stage5C or SMC change. The full legacy suite and cold raw-data reconstruction
are separate from this experiment's scoped verification and remain not_run.
