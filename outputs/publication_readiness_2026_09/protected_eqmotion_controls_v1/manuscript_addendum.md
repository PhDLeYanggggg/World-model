# Matched Public-Predictor Control: Draft Results and Discussion

## Results

We completed the full-EqMotion extension using the previously fitted, pair-site-
excluded producer bank and twelve neural gain/harm heads. Twelve new ExtraTrees
heads used the corresponding neural sample counts as fitting weights. The
observation-eight/prediction-twelve native-step protocol, prediction population,
causal feature interface and intervention rule were shared with the simple-motion
and Transformer controls. No new forecasting model or threshold was selected.

Under strict protection, EqMotion with the neural cost head improved equal-site
available-point ADE over causal constant velocity by 1.609% (nominal site-bootstrap
CI95 0.716--3.039). The forest head improved ADE by 1.263% (0.429--2.735).
The paired neural-minus-forest difference was 0.346 percentage points
(0.217--0.489). Worst site/seed positive-easy degradation was 0.452% and 0.000%,
respectively. Without strict protection, much larger average gains coexisted
with 40.7--47.4% worst site/seed easy degradation.

Protected simple-motion controls limit the forecasting interpretation. With
matched intervention counts and neural heads, damping005 improved ADE by 2.668%
versus 1.609% for EqMotion, with easy degradation of 1.622% versus 0.452%.
The paired difference was inconclusive (-1.059 pp for EqMotion minus damping005;
CI95 -2.361--0.408). EqMotion also did not demonstrate superiority over the
protected Transformer (-0.828 pp; -2.148--0.442).

## Discussion and Boundaries

The results support further investigation of gain/harm estimation, not a claim
that neural forecasts are indispensable. Matching both protection and intervention
counts is necessary: an unprotected simple baseline confounds predictor quality
with intervention control, while a selected count does not guarantee subgroup
safety. The results do not establish a joint-agent or multimodal contribution.

All four SDD sites were exposed during method development. These comparisons are
not independent confirmation; the 3,000-resample intervals use physical sites,
not overlapping windows, and are not multiplicity adjusted. Three seeds do not
increase site-level independence. Histories are offline annotation histories,
with unresolved interpolation provenance. Coordinates are annotation pixels and
steps are native annotation steps, without a verified metric/seconds claim.
The earlier failed EqMotion cost-refit primary comparison remains reported.

This addendum records a new source-development experiment. It does not alter the
frozen manuscript-v2 evidence bundle, reuse external labels for selection, or
claim an independently calibrated deployment policy. All fixed controls,
missing-label bounds and adverse results remain in the accompanying analysis.
