# Exact-Coverage Intervention Control

Date: 2026-09-16. Scope: a missing mechanism control, tested synthetically and at the real past-input boundary. No real predictive comparison, approved policy change or deployment.

## Why The Existing Comparison Was Insufficient

The existing independent and joint policies share forecasts, support masks, a predicted-harm cap and an intervention cap. The cap is an upper bound, not an equality. Joint optimization may select fewer or more agents than independent optimization. A difference in measured error or geometric proximity would therefore combine a change in coverage with a change in the selected agents. This prevents attributing a favorable result to coupling alone.

The initial hand-constructed equal-count example was useful but did not enforce equality on general queries. The new control makes equality explicit for every observable scene query. It does not retrofit a matched-coverage claim to historical results.

## Decision Rule

Let `u_ind(x)` be the existing independent decision from the same past-only inputs and frozen predicted costs. Before any target labels are opened, set `k(x) = sum(u_ind(x))`. Solve the existing joint objective under exactly `sum(u) = k(x)`, preserving the same available-agent mask and predicted-harm cap.

The independent decision is a feasible witness for the exact-count problem when its original solve succeeds. A completed joint solve therefore isolates which agents switch, conditional on the reference's per-query coverage. No second threshold is fitted on future labels to obtain matching.

`select_interventions(..., exact_interventions=k)` is explicitly diagnostic. It may accept a solution with worse predicted objective than the floor because the count constraint excludes the floor when k is positive. It must not replace the ordinary positive-gain policy. `compare_at_independent_coverage` records both decisions, counts, solver validity, zero/nonzero matching and the non-deployment status. `decide_scene(..., include_matched_coverage=True)` exposes this additional comparison while leaving the ordinary five arms unchanged. Formal CLI selection, calibration and confirmation do not silently add this arm or change the protocol.

## Required Interpretation

- Equal counts apply to all agents admitted by past support, before future-label masks are known. Counts within the subset with complete future labels can differ; report those separately instead of claiming conditional equality.
- Matching the mean predicted-harm cap does not match actual harm. This control is not a risk guarantee and not a physical-safety test.
- Zero-versus-zero is recorded as a trivial match, not evidence that interaction modeling helped.
- Solver timeout, invalid binary output or infeasibility returns the floor and marks the comparison unmatched. Do not remove these queries silently or report their floor result as matched. A numerical guard rejects a solver success flag if the returned decisions fail integrality/feasibility checks.
- A matched policy can trade predicted individual gain for lower pairwise proximity cost. Both accuracy and proximity must be scored later; lower proximity alone is not success.
- This is a reference-conditional mechanism ablation. It does not prove that the reference coverage is optimal, that equal counts imply equal compute, or that all risk-coverage operating points are matched.

## Evidence

The authoritative final-version run is [final_version/checks.json](final_version/checks.json). The earlier [checks.json](checks.json) is retained as a superseded pre-final-guard engineering run; it has a different solver hash and must not be used as current-version verification.

On 80 seven-agent synthetic problems, cap-only independent/joint counts differed in 53 cases. Exact-count comparison matched all 80 nonzero cases and agreed exactly with exhaustive feasible enumeration. These are constructed predicted scores, not observed forecast gains or statistical evidence about a population.

The final real-input probe uses random Torch forecast weights and constant synthetic gain/harm scores on the existing canonical reader plus CITR. It checked 61 queries / 618 agents, with 61 count matches, 55 nonzero matches and four queries excluded for missing raw50 past support. It opens no label API and computes no accuracy. Replacing all post-current positions with NaNs leaves inputs, membership, matched decisions and emitted predictions unchanged. Probe timestamps sample the recordings for engineering coverage, not an approved evaluation population. Engineering thresholds and seeds do not authorize scientific choices.

The regression record in [verification.json](verification.json) retains the initial combined-run failure: the solver source was edited while a synthetic final-evaluation resume test was running, causing the provenance guard to reject `Calibration implementation changed`. The implementation was then held fixed and the combined suite rerun. This was not a real training crash, and no provenance guard was weakened to make it pass.

Final fixed-version regression: **224 passed in 62.72 s**, including 20 new exact-count cases. The historical non-hermetic full-suite result (1,870 pass / 1 unrelated failure) was not rerun or replaced by this focused result.

## Scientific Status

The source inventory was rechecked: the local top-down roots remain OpenTraj and SDD. CITR contributes one controlled physical site, not independent-scene replication. DUT raw acquisition, source-use approval and independent confirmation remain unresolved. CREATE access has no new credentials or project-path evidence; no remote job was started.

The main temporal protocol, data roles, aggregation and risk budgets are still awaiting the previously requested decisions. Real model fitting and this matched-control forecasting experiment are `not_run`, not failures or successes. No new ADE/FDE lift, baseline superiority, calibration guarantee or CVPR readiness is claimed. Raw frames and dataset-local coordinates remain non-metric/non-seconds claims. Stage5C and SMC are off.

The next real comparison must predeclare this diagnostic, retain the ordinary deployment controls, evaluate matched and unmatched query counts, and report paired whole-scene uncertainty with the approved data roles. It must also include a credible cost-sensitive deferral comparator; the [literature boundary](../joint_intervention/deferral_and_coverage_prior_work.md) explains why gain/harm routing alone is not an established novelty claim.
