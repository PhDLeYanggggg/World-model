# Bound-Aware Cost Learning: Failure Analysis

## Material Passport

Analysis of the completed, separately replayed SOURCE experiment. All numerical
claims refer to `summary.json`, `verification.json`, fitting receipts and
`selected_risk_failure.json`. No new fit, private calibration reuse or final
confirmation is created by this analysis.

## What Changed And What Did Not

The experiment changes supervision on wholly unknown outcomes. It does not
change the fixed forecasts, causal feature schema, head architecture, initial
weights, normalization, sampling or complete training budgets. Both arms retain
unknown rows as past-only context; only the bound-aware arm supervises their
conservative moment contributions. Query-average supervised support changes
with that treatment and is part of the registered intervention.

The full-bound fitting objective falls from 4.883677 to 1.489479. The known-only
objective instead rises from 0.901362 to 1.011491. The repair therefore addresses
a real fitting mismatch but does not uniformly improve cost regression. These
two columns are different targets, not interchangeable training-loss rankings.

## Three Distinct Outcomes

1. **Observed mean utility:** the paired known-label effect is +0.3774 pp with
   95% locality-bootstrap interval [-1.4814, +1.9475]. It is inconclusive.
2. **Registered primary:** the direct paired completion-lower effect is
   -1.7031 pp [-5.7081, +1.1166]. It does not support advancement. Its negative
   point is not a significant observed-ADE regression.
3. **Full recording risk:** supported views increase from 28 to 108 out of 444,
   but neither policy passes. Known-all violations increase 79 to 143, whereas
   known-easy violations decrease 330 to 235. This is a mixed trade-off.

The separate individual lower gains, 1.0579% and 2.7669%, cannot be subtracted
to replace the direct paired primary: their unknown-row uncertainty sets do
not cancel in that way. Secondary binary-policy gains do not rescue the primary.

## Where The Learned Guard Fails

The worst signed easy-net bound-aware view is SOURCE locality 008, recording 77,
`single1_seed29_critic0`. Its predicted selected easy-risk ratio is 0.059779%,
but observed selected easy positive-harm risk is 57.7637%; the unknown-completed
upper risk is 96.6815%. Signed easy-net degradation is 43.6929%. This view has
381 active rows out of 421, including 11 active rows with wholly unknown labels.

Both the predicted numerator and denominator can be wrong. Their ratio contrast
does not identify either component as the sole cause. It does establish that
the learned pointwise predicted check is not a calibrated guarantee for the
actual recording selected by that same rule. Conditional mean algebra and
lower fitting loss do not bridge that gap.

Four of twelve localities have negative paired known effects. Locality 082 is
especially problematic: -7.0221 pp known and -21.1166 pp paired completion lower.
The bound-aware arm has 1.0946% known floor-relative gain there versus 8.1167%
for control. Retain this locality and all recordings; do not select the other
eight sites or a favorable seed as the headline result.

## What The Experiment Rules Out

- Exact full inference and metric replay, paired initialization/draw checks and
  first-job full refits do not indicate an unfinished run or accidental cache
  mismatch. They do not prove independent scientific confirmation.
- Explicit bound supervision helps its intended fitting target. Its omission
  was not the only remaining reason for selected-risk failure.
- Attenuation is not automatically calibrated risk control. The same fixed
  2% predicted guard still underestimates actual selected easy harm.
- More supported views are not a complete safety pass, and a better global
  easy-net mean does not establish every-record easy preservation.

## Next Decisive Repair

Before another architecture or loss sweep, decompose predicted versus realized
H, EH, R and ER on **the same selected sets**, separately on critic fitting
sites and sites excluded from both fitting stages. Compare selection-conditioned
error with fixed-bank error and label-support uncertainty. Keep the fixed
reference rule, original 2% contract and all SOURCE records. This will separate
poor conditional cost fitting from new-locality shift and denominator/support
error; it must not be replaced by inspecting another selected-risk ratio alone.

Any subsequent uncertainty-aware cost or richer past-only representation trial
must be registered before fitting. No threshold is chosen from these outcomes,
and independent confirmation must not be opened to repair this SOURCE failure.

## Limits

The 444 recording/pipeline views repeat 74 recordings across twelve exposed
SOURCE localities and three actual seeds. They are not 444 independent sites.
Nested exclusion prevents fit overlap for each view; it does not erase SOURCE
development exposure. Available-label ADE is not full12 ADE; conservative
bounds cover wholly unknown rows, not missing steps inside partial futures.
Image-local pixels/raw stride12/released silver are not metric, seconds or
human gold. This is cost learning, not new neural dynamics, physical safety,
true 3D or foundation evidence. Deployment is unchanged; Stage5C and SMC are off.
