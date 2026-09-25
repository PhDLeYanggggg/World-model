# Failure Analysis

Status: completed source-development experiment; no promotion. Predictions and
labels are image-pixel obs8/pred12 raw-step quantities, not metric or seconds.

| Hypothesis | Evidence | Interpretation |
|---|---|---|
| Training-constant features explode outside fitting localities | Preceding diagnostic sampled up to 512 evenly spaced rows per locality for each of 54 frozen heads; no constant-feature excursion exceeded one standardized unit | Not supported on that sample; not a proof that every input is well behaved |
| Smaller OOF producers alone repair transport | Previous frozen-head diagnostic retained both smaller producers and all matched controls; replacement was inconsistent | Producer identity is not a sufficient repair; other shifts remain possible |
| Cost magnitude is unconstrained | New causal envelope gives a valid bound even under partial future-label support | Mathematical constraint implemented and tested; not a calibration guarantee |
| Better utility alone establishes neural advantage | Utility-only: 0/18 positive neural ADE points versus matched damping; 17/18 negative intervals | No support in this experiment |
| Better risk parameterization safely releases useful neural predictions | Risk-only: four positive neural ADE intervals; both: two; all six unsafe | Some useful predictions are released, but intervention safety remains unresolved |
| Low expected positive harm remains reliable after selection | Locality 110 example has predicted event harm/reference about 0.69%, observed about 47.09% | Selection-conditional underestimation is directly observed; causation is not uniquely identified |
| Average easy accuracy suffices | Several positive contrasts improve easy averages yet harm 2-4 zero-CV-error rows | Retain separate zero-reference support; average gains cannot erase these harms |
| The neural predictor simply has no opportunity | More gain is captured after risk-head replacement, but with excess paid harm | There is forecast opportunity, not a demonstrated deployable controller |

## Selected-Harm Example

This slice is descriptive and was identified after the fixed readout. It is
not a validation-selected model, new threshold, or independent confirmation.
For fold2/seed17/easy on locality 110, selected-row event moments are:

| Arm | Predicted reference mass | Predicted positive harm | Observed reference mass | Observed positive harm |
|---|---:|---:|---:|---:|
| Old | 0.546934 | 0.0081933 | 0.613057 | 0.196514 |
| Risk only | 0.515923 | 0.0036177 | 0.604998 | 0.284539 |
| Both | 0.515223 | 0.0035641 | 0.592504 | 0.278996 |

The both-head arm captures 10.2580% of CV error mass as positive gain, pays
3.2320% as harm, and nets 7.0260% on this locality. Its easy mean ADE increases
from 1.8572 to 2.1849 pixels, a 17.6432% degradation. Capturing useful predictions
and violating easy preservation occur simultaneously. These moment ratios and
net degradation measure different quantities and must not be interchanged.

At overall-view level, intervention increases from 1.2889% to 37.5189%, and ADE
gain over CV from 0.1143% to 3.2399%. The new forecast is unchanged: these are
decision changes, not new dynamics learning. The complete other seeds, folds
and damping controls remain in [results](results.md).

## Loss and Representation Limits

All new heads use matched native-unit MSE, scaled by the same fitting-only CV
error constant. Equal samplers and parameter counts isolate parameterization
better than an unmatched architecture comparison. However, neither batch MSE
nor a geometric upper bound optimizes calibrated tail risk under a learned
selection rule. Easy-event positive harms are sparse and candidate dependent.
The current experiment does not distinguish event-probability error from
conditional-severity error, support extrapolation, or limited causal features.

The loss plot is a training diagnostic. It does not establish downstream lift;
different tasks have different targets, so their absolute loss magnitudes are
not directly comparable. No future label mask or endpoint enters inference.

## Next Falsifiable Repair

Use only opened source-development roles to separate event support from
conditional positive-harm severity, and test whether the selected subset's
harm is better predicted. Keep the same forecast producers, fitting budgets,
matched damping and old-controller controls. Keep zero-CV harms separate from
relative-risk denominators. Test a single registered change before scaling it.

Do not select the favorable arm above, tune the 2% limit on these outcomes, or
open reserved confirmation to repair this controller. New score functions
need their own source-excluded calibration; old maps are not transferable
safety certificates. A future successful source repair would still require
independent scene evidence before a deployment or paper claim.
