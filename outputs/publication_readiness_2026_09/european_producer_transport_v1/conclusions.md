# Producer Replacement Does Not Repair Neural Intervention

## Result

I completed the frozen-head producer-transport diagnostic. Replacing a final
four-locality predictor with a two-locality predictor does not consistently
improve the protected policy. This rejects a simple replacement as the repair;
it does not rule out all forms of producer shift or better retrained heads.

There are 18 fresh inference banks, 3,827,628 predictor-row pairs, 72 fixed
policy views and 36 small-versus-full comparisons. No model was trained,
calibrated or selected in this experiment. Eighteen distinct raw-predictor
comparisons are repeated under two event gates; they are not 36 distinct raw
experiments. Every registered fold, seed and producer half is retained.

| Same-row contrast | Comparisons | Positive point estimates | Strictly positive conditional CI | Strictly negative conditional CI | Gain range (%) |
|---|---:|---:|---:|---:|---:|
| Small vs full, protected all ADE | 36 | 9 | 5 | 15 | -3.0360 to +0.9573 |
| Small vs full, protected hard ADE | 36 | 11 | 5 | 15 | -3.4442 to +0.9074 |
| Small vs full, protected easy ADE | 36 | 13 | 2 | 9 | -1.2361 to +0.5381 |
| Small vs full, raw ADE, distinct pairs | 18 | 6 | 0 | 8 | -10.8544 to +3.4851 |

Each interval uses 3,000 resamples of the same eight excluded source localities
for that pair. All views share twelve opened development localities. Intervals
are conditional, not independent confirmation, and are not corrected for
selecting among many comparisons. No winner is selected here.

## Safety Is Not Repaired

| Producer | All-event views passing observed safety | Easy-event views passing observed safety |
|---|---:|---:|
| Full four-locality predictor | 5/9 | 4/9 |
| Half 0 predictor | 5/9 | 4/9 |
| Half 1 predictor | 5/9 | 5/9 |
| Matched damping 0.97 control | 7/9 | 9/9 |

Observed safety means worst-locality net easy degradation <=2% and no added
error on observed zero-CV rows. It is not a future-risk guarantee. These
per-fold eight-locality counts must not be compared directly with the previous
study's pooled, rotated counts.

Of the five strictly positive all-ADE intervals, only two belong to views that
pass observed safety. Their gains over the full-producer policy are 0.03335%
and 0.02465%, from different fold/seed combinations. The larger improvements
in fitting fold 1 fail safety. This is not a reproducible safe neural advantage.
Replacing full4 with half1 changes only one safety failure into a pass; the
other paired safety decisions remain unchanged. Worst easy degradation among
the small-producer views reaches 6.4125%.

## What the Diagnostic Separates

**Producer/source variation is real, but matching producer size is insufficient.**
In fitting fold 1, half0 improves raw ADE while half1 substantially worsens it
for every seed. Both use two fitting localities. The replacement also changes
the producer's fitting roster, normalizer and potentially its training-selected
motion floor. It cannot identify a pure effect of training-set size.

**The score heads still miss intervention harm.** For the all-event views,
selected event-harm ratios exceed 2% in 43/64 defined locality/view cells for
full4, 44/64 for half0 and 33/59 for half1. These are selected positive-harm
ratios, not net easy degradation or independent replications. Smaller producers
do not solve the selected-population reliability problem. All locality means,
errors, empty slices and undefined ratios remain in the supporting tables.

**Better raw motion is not automatically captured by the controller.** In
fitting fold 2, full4 has raw ADE gains of 9.56--9.76% over CV, while its
all-event controlled gains are only approximately 0.00005--0.1037%. The full
table retains the corresponding easy, hard, FDE and intervention-rate results.
Hindsight opportunity does not prove that past observations can identify it.

## Next Repair

The next controlled change should target candidate-specific gain/harm learning,
not choose the favorable half or loosen thresholds using these outcomes.
First compare training and excluded-row sensitivity to the candidate rollout,
normalization and output saturation on the frozen heads. Then register one
source-only refit that exposes candidate-relative trajectory differences more
directly, retaining the old head as a matched control. Hold the forecaster,
sampler, 2% budget and excluded-locality boundaries fixed. Check selected harm,
missed benefit and every seed, not just population regression error.

This is the next hypothesis, not a demonstrated repair. Independent model
selection, risk calibration, confirmation and DroneCrowd readout stay closed.

## Evidence and Limits

- `fresh_run`: new inference, paired metrics, opportunity/score diagnostics,
  full metric recomputation, raw ADE/FDE accounting and scoped tests.
- `cached_verified`: source arrays, split roles, 18 inner and nine full
  forecasting checkpoints, 54 gain/risk heads and original decision banks.
- `not_run`: new training, threshold refitting, independent confirmation,
  new joint-controller optimization, remote M3W asset inspection and deployment.

All 18 checkpoints reproduce their first 4,096 held-index predictions exactly,
not a random sample or a full regeneration of every bank. All 72 original
decision receipts reconstruct, unchanged full4/damping score arrays match,
and all 72 view metrics reproduce. There are 201 passing tests in 30 scoped
files, not a run of the full legacy suite. The plotted intervals were visually
checked. Bank inference took 582.22 seconds, excluding metric/replay/report work.

These are released detector tracks in image pixels, observed8/predicted12 on
raw stride12. They are not historical t50, verified seconds, metric, human-gold
annotations, physical safety, true3D or a foundation model. Historical Stage37
exploratory evidence is not recertified by this experiment. No deployment
promotion, Stage5C execution or SMC. The project is not submission-ready.

See [all fixed comparisons](results.md), [raw trajectories](raw_forecast_results.md),
[score reliability](score_reliability.md), [opportunity accounting](opportunity_attribution.md),
[failure taxonomy](failure_taxonomy.md), [gates](gates.md) and
[completion checks](completion_checks.json).
