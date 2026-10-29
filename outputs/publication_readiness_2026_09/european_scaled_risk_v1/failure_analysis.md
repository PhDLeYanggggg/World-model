# Why Fixed TRAIN Budget Scaling Does Not Advance

## Decision

The registered candidate fails. No deployment, independent transfer, Stage5C or
SMC is authorized by this result. All 144 planned fits and 72 evaluation views
completed; failure is scientific, not an unfinished or downgraded run.

## What the Intervention Changed

The prior TRAIN-only gradient audit found median easy-penalty/data gradient ratio
0.000698029. The repair divides both empirical risk excesses by positive, fixed
TRAIN reference-budget scales in the primal penalty and dual update. All other
registered settings stay fixed. The 72 fresh unit controls exactly reproduce
their old weights, optimizer, duals, draws and traces.

This tests optimization conditioning/effective penalty capacity. Positive
scaling preserves the TRAIN constraint zero set; it does not prove equivalence
to the hard decision or the reported pooled-agent risk.

## Evidence-Supported Taxonomy

| Finding | Observed evidence | What it does not establish |
|---|---|---|
| TRAIN optimization improves | Soft easy-excess views61 to12; all-excess38 to13 | Whole-TRAIN feasibility or generalization |
| Development safety does not improve on known outcomes | Easy-risk violations58/72 in both arms | Which one mechanism caused failure |
| Cost prediction degrades | Signed-score MSE delta+2.370133, nominal95% CI[0.797264,4.891259] versus unit control | A universal architecture limitation |
| Task performance is not retained | Paired lower utility-1.890975pp; same-count-0.484529pp versus unit control | An identified exact utility difference under missing outcomes |
| Harm remains badly underestimated |66/67 defined selected-easy ratios below1; median0.040247 | Independent calibration or physical safety |
| Soft/hard discrepancy remains | Mean development soft action0.113384; hard rate0.235609; diagnostic disagreement0.188633 | Calibrated switch probabilities |
| Support is still missing |67 upper violations,5 undefined easy ratios,0 finite supported full-policy views | That undefined views are safe |
| Aggregation is not equivalent | Two scaled TRAIN hard-monitor views pass query-mean easy but fail agent mass | Explanation of all58 development violations |

The full-policy paired lower contrast versus unit control is negative in11/12
localities; the same-count contrast is negative in9/12. These are descriptions of
exposed development results, not new confirmatory subgroup hypotheses. The
candidate selects112,749 row occurrences versus109,401 for the unit control,
yet its equal-view mean intervention is lower. Those summaries use different
weights; neither can be substituted for the other.

## Why More Penalty Is Not the Demonstrated Fix

The original easy-gradient weakness was real, and scaling substantially reduces
its empirical TRAIN violations. However, the scaled candidate still fails on
development recordings and loses on the registered conservative utility metric
even under equal intervention counts. A smaller action count alone is therefore
not an adequate explanation. Only1/72 scaled final duals reaches the cap;
universal cap saturation is not supported either.

The fixed-monitor aggregation diagnostic finds hard easy excess in3/72 views
under query means and5/72 under agent mass, with9 undefined easy-reference
cohorts. The discrepancy is a real objective-contract issue but modest on this
monitor. These TRAIN cohorts differ from development cohorts and exclude unknown
labels, so their counts cannot causally explain the development risk gap.

## Boundaries and Next Experiment

1. Align the TRAIN risk estimator with the unchanged agent-mass readout, keeping
   the original estimator as an exact matched control. First verify the sampler
   weighting and zero-reference behavior; do not quietly alter the risk budget.
2. Diagnose recording-held-out harm prediction using TRAIN-only internal folds,
   support/coverage slices and fixed soft/hard decisions. Keep unknown outcomes
   explicit. Test whether the problem is extrapolation, rare-harm estimation or
   decision-surrogate mismatch before choosing another model size or penalty.
3. Register one supported repair, retaining unit, quadratic and utility controls.
   Do not use the independent selection/calibration/confirmation roles to rescue
   failed development performance. Any independent inference remains not_run.

The preceding three items are next work, not completed experiments. No new
trajectory dynamics, joint-agent contribution or cross-domain success is shown.
The current scaffold remains2.5D; units remain dataset-local/raw-frame. Silver
labels are not human gold. Historical Stage26/37 results remain exploratory.

## Provenance

[Fixed readout](readout/conclusions.md), [TRAIN gradient audit](../european_task_risk_gradient_v1/conclusions.md),
[TRAIN aggregation audit](../european_risk_aggregation_v1/conclusions.md).
Readout summary SHA256:
`643f36a72529968422dff8db901dccd5a4da09b5db803a9d6f87126dead1d153`.
