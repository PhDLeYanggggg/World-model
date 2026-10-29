# Failure Analysis

## Supported Findings

1. **Average fitting and selected-region estimation diverge.** Estimation-only
   improves eligible all/easy MSE by80.29%/55.65% versus the original, yet original-
   selection easy MSE does not improve. This is not just a cross-scene problem:
   the failure exists on fitting data under a fixed causal selection.
2. **Easy-harm underestimation remains.** Predicted normalized easy-harm mass on
   the fixed selection is2.268195 versus95.560642 observed. The near-original
   MSE is therefore not evidence of accurate low-risk estimates.
3. **Coverage expands without risk support.** Selected occurrences rise from
   62,902 to419,270, with62/72 known easy-risk violations and0/72 full support.
   The new model cannot be promoted for producing more positive-looking actions.
4. **More updates alone are not a repair.** The equal-budget coupled continuation
   also fails complete support. Estimation beats it on MSE, but that comparison
   does not excuse failure against the original or the fixed12/18 screen.
5. **The effect is not uniform.** Five of six context aggregates have slightly
   worse fixed-selection easy MSE versus original. Seed43 improves slightly;
   seeds17/29 do not. All twelve source groups include known easy-risk violations
   under estimation. See[slices](slice_diagnostics.md), described after readout
   and never used for new model selection.

## Unresolved Explanations

The fixed decoder's conditioning, insufficient conditional input information,
averaged-loss emphasis, optimizer history and selection-induced optimism remain
plausible. This contrast demonstrates that removing both action terms does not
solve them; it does not identify one of these as the unique cause. Similar
near-zero MSE changes should not be described as large causal effects.

No held generalization inference was performed. Failure on fit is sufficient
to reject this particular routing proposal, not every conditional risk estimator.
No missing-label filtering, empty-cohort shortcut, increased2% budget or adaptive
test threshold is permitted. The intervention-count matched result is also
negative, so reduced coverage alone is not the next scientific claim.

## Statistical Interpretation Audit

Eleven concerns checked; these are exploratory fitting results, not hypothesis-
test significance or an external safety guarantee.

1. Group reversal: context,source and seed slices reported; no uniform-lift claim.
2. Ecological inference: view means do not establish per-agent correctness.
3. Selection bias: the original selected cohort is fixed, causal and explicitly
   conditional; its score is not a whole-population score.
4. Collider concerns: conditioning on selected actions limits interpretation;
   no causal population mechanism is inferred from selected-row associations.
5. Base rates: known/unknown,eligibility,selected and violation denominators are
   all reported. A count of violations is not an event prevalence estimate.
6. Regression to mean: original and equal-budget coupled controls are retained;
   no worst-slice before/after improvement is presented without comparison.
7. Survivorship: every one of36 fits and72 views is retained; unknown outcomes
   remain visible and are bounded in support accounting, not silently dropped.
8. Look-elsewhere: all nine fixed conditions and failures are reported; no
   favorable-source or seed selection and no confirmatory fitting CI.
9. Forking paths: matched arms and budgets frozen before training; posthoc slices
   are descriptive and do not revise the gate.
10. Causal claims: a controlled computational loss contrast supports statements
    about this implementation, not real-world physical or clinical causation.
11. Reverse causality: labels are supervision/evaluation only; future outcomes
    cannot determine inference features, eligibility or the fixed selection.

## Next Discriminating Work

Do not repeat threshold-only or clipping-only searches. Reconcile the existing
OOF,selected-risk,hurdle and tail controls before registering a fitting-only test
of whether the current causal representation can distinguish rare harm within
the frozen selection. Separate approximation error from selection optimism;
do not mistake another mean-MSE reduction for policy evidence. Any new held
design must be frozen separately, and independent confirmation remains reserved.
