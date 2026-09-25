# Failure Taxonomy: Calibration of Delivered Policy Risk

## 1. Event/Reference Mismatch Was Repaired, Not Assumed Away
An earlier calibration helper used CV for both easy-event membership and the
harm denominator. That is not the present question. The new tested helper
anchors membership to A's positive-CV easy cut, but uses actual delivered R
error in the harm denominator. This implementation repair was made before
registration and fitting. Residual transport failure cannot simply be blamed
on that old denominator mismatch.

## 2. Population Moment Bias
Across 108 dependent full-neural locality/settings, the raw predicted-to-actual
selected-harm ratio spans 0.134396-0.593707, median 0.300808. Reference mass
bias spans 0.893919-2.296260, median 1.494471. Thus the numerator tends to be
too small and denominator too large on matched outcome support. This is an
observed score-calibration failure, not merely a confidence-labeling issue.

Source-C population scaling increases harm factors by 1.166744-3.495589 for
the full neural pair and can reduce denominator factors to 0.642234. On new
opened localities, its adjusted selected-harm bias still spans 0.182006-1.481448
(median 0.541182). A single population correction does not make the selected
tail, each conditional event or each locality calibrated.

## 3. Conditional Easy Risk Is Hidden by Aggregate Success
Rescaling makes all-event positive harm pass in every full-neural locality/view,
while 38/108 easy-event views still exceed 0.02. Net easy degradation is zero.
The fixed selected-risk grid has 52/108 easy violations despite 18/18 C fits
being feasible and all selection settings preserving net easy error.
Do not replace a positive-part harm constraint with net error after the result.

## 4. Geography of Failure, Not a New Tuning Rule
For the full neural grid, easy violations by readout locality are:

| Locality | Violating settings / 18 | Easy positive-harm ratio range |
|---|---:|---:|
| 087 | 11 | 0.006142-0.049511 |
| 092 | 5 | 0.003563-0.032184 |
| 093 | 11 | 0.009624-0.078996 |
| 103 | 4 | 0.004442-0.027525 |
| 104 | 6 | 0.009015-0.033667 |
| 125 | 15 | 0.005590-0.045466 |

This is a posthoc description. No locality-specific threshold was fitted from
these outcomes. Locality 125 is frequent, while 093 has the largest observed
ratio; neither alone explains the failure across all settings.

## 5. Small C Rosters Do Not Establish Transfer
Fitting on three C localities and checking the fourth yields 59/72 feasible
full-neural views, 60/72 motion-neural and 66/72 for each ridge pair. This is
already imperfect before reaching the six selection localities. The held-C
diagnostic did not select the final four-C map. It supports instability as a
working hypothesis, not a proof of which causal visual/domain factor shifts.

## 6. Coverage/Accuracy Cost
Full neural rescaling reduces intervention from 26.08%-42.32% to 12.85%-39.62%.
Every seed-mean all-ADE interval versus the raw rule is negative. The grid
retains 16.19%-40.18% coverage but still fails complete risk in 18/18 settings.
Reference-only fallback is structurally risk-free relative to itself, not a
solution proving useful neural prediction. No threshold is relaxed to hide this.

## 7. Attribution Remains Unresolved
This study reuses fixed predictors, so it cannot demonstrate new learned
dynamics. Ridge also improves its risk counts with rescaling and still fails
some settings. Previous matched-coverage attribution did not show stable
neural ranking superiority. The evidence does not justify a larger backbone
as the first repair or selecting the most favorable neural seed.

## 8. Statistical and Label Limits
Six reused localities, correlated windows and 848 unknown futures cannot
justify independent calibration. Unknown rows remain in inference; support-
matched diagnostics avoid treating them as zero error. The harm ratio is not
known bounded in [0,1], so a bounded-loss theorem cannot be attached unchanged.
Detector-derived labels and fixed query sampling also limit generalization.

## Next Repair and Its Boundaries
Use source-only held-roster diagnostics to locate unsupported causal feature
regions and selected-tail moment errors. Compare an explicit support-aware
fallback with a matched ridge control and a scene-query aggregate controller;
retain the same delivered-reference targets and both harm events. Retrain or
freeze a new rule before any next readout. Any source-local success still needs
independent calibration feasibility and later confirmation. This next repair
is not_run here. No new scene is opened, no test threshold is tuned, no Stage5C
or SMC executes, and no physical/metric/seconds/foundation claim is made.
