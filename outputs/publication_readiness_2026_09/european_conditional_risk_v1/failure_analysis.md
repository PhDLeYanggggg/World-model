# Failure Analysis

The experiment separates observed results from explanations that still need a
new controlled test. It does not relabel source-development results as a test
set or change a threshold after comparative readout.

| Failure or limitation | Evidence | Interpretation and next test |
|---|---|---|
| Utility loss under risk gating | Easy-neural pointwise ADE gain is 0.17--0.43% vs CV, down from the preceding policy's 4.18--4.43%; intervention drops to 0.29--1.25% | Protection largely reduces intervention. Test protected simple forecasts at matched risk and coverage before increasing capacity. |
| Easy-event target is only a partial repair | Matched guarded easy-versus-all neural contrasts are positive for all seeds; unguarded seed 17 remains uncertain | Event weighting matters within this model, but does not establish superiority over previous policies or simple controls. |
| Neural head is not essential yet | All-event ridge gives 0.48--0.61% gain with observed pointwise safety | Retain ridge in every next comparison; do not claim a neural contribution from architecture alone. |
| Joint budget can spend harm unevenly | Unguarded easy-neural joint worst-locality easy degradation is 2.1034% / 2.0391% in seeds 29 / 43, despite acceptable full-pointwise results | Query-level predicted mass is not a locality-level realized guarantee. Test budget allocation separately from prediction. |
| Interaction contribution unproven | Exact-count joint contrasts are zero, tiny or unsupported | Do not infer a graph contribution from joint-pilot gains over CV alone. Pair-proximity proxies are not physical collision labels. |
| Zero-event support sparse | Four zero-CV cases, all locality 008; outer fold 2 has no fitting zero events | Guard abstains throughout the unsupported fold, not selectively on identifiable difficult agents. Presence of a few events in other folds is not certification. |
| Partial labels limit safety interpretation | Those four cases have only 2/12 labels and no final endpoint; joint pilot has none | Observed zero ADE does not imply a correct whole future. Retain missingness and never filter by unseen labels at inference. |
| Event risk is not calibrated | Both predicted denominator and harm are regressions; denominator loss is symmetric, harm underestimation weighted 4x | Their ratio need not bound actual conditional risk. Independent calibration remains not_run. |
| Producer-size mismatch remains | Cost targets use four-source-locality producers, held predictions use eight-source-locality producers | Possible cost-distribution shift, not quantified causal blame. A versioned matched-producer experiment is needed. |
| Weak scientific independence | All 12 source localities have informed development; cross-fit training chains exclude held localities but source results have been inspected | Bootstrap uncertainty is conditional. Reserved selection/calibration/confirmation roles stay closed. |

## Numerical Defect and Bounded Repair

Ten recorded views, corresponding to five seed/query cases with and without
the support guard, returned `solver_not_optimal_floor`. They are singleton
queries in locality 020. Predicted error mass was approximately 2.67e-21 to
2.58e-15, while positive predicted harm made the ratio enormous. The existing
solver path retained these infeasible variables until numerical optimization.

For nonnegative costs, any candidate whose individual cost exceeds the whole
budget cannot occur in a feasible subset. The new helper prunes that candidate
before division. It does not increase epsilon, relax the risk budget, inspect
future labels or change the original mathematical feasible set.

All ten targeted real-case replays solve optimally and preserve the original CV
decision. Tests additionally compare all subsets of 80 five-agent problems,
2,560 subset checks, and cover missing support and the tiny-denominator case.
See [numerical_audit.json](numerical_audit.json). This helper is separate from
the immutable V1 code. There is no retroactive replacement of the frozen
analysis, nor a claim that every multi-agent solver call has been rerun.

Consequently, these numerical failures do not explain the loss of pointwise
utility: the repaired cases still cannot afford intervention. Better numerical
conditioning is an implementation repair, not new predictive evidence.

## Prioritized Repair

1. Register a matched risk-protected simple-motion versus neural comparison.
   Hold roles, forecasts, support, samples and risk budget fixed within each
   comparison. Report all seeds, equal-locality accuracy, worst-easy, zero-CV
   harm and actual intervention counts. Do not choose a winner on these sources.
2. Diagnose denominator/harm calibration and producer-size shift using source
   folds only. Distinguish inability to estimate risk from absence of forecast
   headroom. Preserve unknown-label targets in the policy population.
3. Advance to reserved-role calibration only after a concrete source-supported
   method is frozen. Rare-event support and the number of independent localities
   may prevent a tight safety claim; report that limit rather than weaken it.

No deployment promotion, Stage5C, SMC, metric/seconds, true-3D, foundation or
submission-readiness claim is supported by this experiment.
