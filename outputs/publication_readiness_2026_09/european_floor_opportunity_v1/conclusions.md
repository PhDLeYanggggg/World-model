# Default-Action Mismatch Hides Incremental Neural Value

## Result Passport

- Fresh: 36 frozen-policy diagnostic groups, eight support slices, error decomposition and 3,000 paired locality-bootstrap draws per view.
- Cached and hash-verified: both cross-moment model versions, trajectory forecasts, utility/risk heads, splits and saved causal decisions.
- Not run: new training, risk recalibration, independent confirmation or new CREATE training.
- Decision: meaningful development signal, not a safe deployment or submission-ready result.
- Registered and pushed before new readout: `77e610eb`.

The preceding study found every neural all-ADE comparison inferior to equally
protected damping. That conclusion about the original policies reproduces.
However, it did not distinguish neural errors from a default-action mismatch:
the neural policy reverted to CV when vetoed, whereas its comparator could still
use protected damping. This diagnosis holds all neural decisions fixed and
changes only that default action in an explicitly offline comparison.

## Controlled Result

Each row contains nine fold-seed comparisons. The event is the risk head target,
not the evaluation subset. Ranges are direct relative improvements over the same
protected-damping policy, not differences between two CV-relative scores.

| Mode / event | Rebased all-ADE gain (%) | Positive all intervals | Hard-ADE gain (%) | Positive hard intervals | FDE gain (%) |
|---|---:|---:|---:|---:|---:|
| Batch / all | 0.3435 to 2.0713 | 9/9 | 0.3279 to 1.6359 | 9/9 | 0.4683 to 2.9483 |
| Batch / easy | 0.1270 to 0.3303 | 9/9 | 0.0117 to 0.0813 | 9/9 | 0.1274 to 0.5661 |
| Fitting / all | 0.3231 to 2.0317 | 9/9 | 0.3067 to 1.5823 | 8/9 | 0.4555 to 2.8450 |
| Fitting / easy | 0.1278 to 0.3768 | 9/9 | 0.0233 to 0.0911 | 8/9 | 0.1262 to 0.6223 |

All 36 all-ADE and FDE intervals are positive; 34/36 hard-ADE intervals are
positive and two include zero. These are dependent, unadjusted development
intervals, not 36 independent replications or a guarantee in unseen scenes.
The original policies remain -3.0628% to -0.1497% versus protected damping.
The rebased diagnostic is +0.1270% to +2.0713%. No seed or model was selected.

Average improvement does not mean every locality improves. Two rebased views
have a negative worst-locality all-ADE point (worst -0.0117%), and seventeen have
a negative worst-locality hard point (worst -3.0515%). The largest locality
all-ADE p95 increase is 0.8680% relative to the floor's p95. Every locality and
tail value is retained in the aggregate CSVs, rather than hidden by the mean.

The floor/neural oracle has 12.8927% to 23.9804% attainable benefit. It uses
future labels and is not a model. Existing masks capture only 0.595% to 22.039%
of that available benefit, depending on the head. Thus both unused prediction
value and controller limitations remain; the forecasts are not proved useless.

## Where the Difference Comes From

The exact per-row accounting is:

```
original gain = captured neural benefit - selected harm
                - CV fallback regression + CV fallback relief
rebased gain  = captured neural benefit - selected harm
```

Every component is summed per locality, divided by that locality's protected
floor error, then averaged equally. The difference between original and rebased
gain is 0.3047 to 4.0196 percentage points. This is an additive decomposition,
not the direct percentage improvement over the original policy; both are saved.

For batch/all heads, CV regression is 3.6048--4.7948 pp while CV relief is
0.3184--1.0546 pp. Captured neural benefit is 0.6112--3.0141 pp and selected harm
0.2384--0.9428 pp. Their original-policy deficit is therefore not explained only
by inaccurate neural trajectories or too many bad switches. A useful fallback
action is also being discarded. Fitting/all shows the same direction.

Missed benefit is concentrated after nonpositive-utility and risk-veto decisions.
For batch/all, these account for 2.8256--8.0117 and 6.1509--16.5180 pp respectively.
This does not justify releasing those vetoes: the accounting looks at realized
future errors after the decisions. It is not an inference-time harm guarantee.
The min(CV, floor, neural) oracle is reported separately because it also benefits
from undoing harmful damping, not just selecting the neural forecast.

## Safety Still Fails

All 36 rebased views meet the observed positive-easy degradation limit. Worst
degradation is 0.5070% for batch and 0.7221% for fitting, below 2%. But twelve
views in each mode still harm zero-CV cases; the six remaining views in each
mode contain no zero-CV cases. Observed preservation is only 6/18 per mode.

The actual zero-reference evidence consists of four rows from one locality,
each with two future labels and no endpoint. Repetition across folds, targets
or seeds is not additional independent support. This small, incomplete set
neither validates full-horizon safety nor permits deleting the observed harm.
Changing the fallback cannot repair those selected neural forecasts: their
switch decisions and trajectories are unchanged.

The risk heads were learned relative to CV, not to the protected floor.
Rebasing the default action does not recalibrate their risk predictions. The
result is therefore a useful design diagnosis, not a newly safe model.

## Annotation Sensitivity

The unique indexed population contains 318,969 rows from twelve already-opened
development localities: 193,705 have all twelve future labels, 118,217 have
partial labels, and 7,047 have no future label. These are overlapping windows,
not independent samples. Unknown rows remain in the population and are not
counted as zero-error successes. Fixed eight-locality rosters remain in every
fold and subset, even if a metric would be undefined.

Rebased all-event ADE gains remain positive in both complete and partial slices:

| Mode | Complete-label gain (%) | Partial-label gain (%) |
|---|---:|---:|
| Batch / all event | 0.4619 to 2.7500 | 0.2319 to 1.4871 |
| Fitting / all event | 0.4684 to 2.6579 | 0.1869 to 1.4977 |

All nine intervals in each table cell's family are positive. The easy-event
versions also have positive intervals in both slices. Completeness is used only
for evaluation sensitivity, never for inference or replacement of the primary
population. This narrows, but does not eliminate, annotation-bias concerns.

## Next Controlled Fit

Keep the trajectory candidates and strong floor frozen. Learn incremental
benefit and harm against that floor rather than against CV, using source-cross-
fitted floor predictions. Exclusion must cover the entire producer chain;
in-sample floor scores must not silently become honest training targets.
Explicitly diagnose zero-event support and add no future-label gate. Keep the
same 2% constraint, uncertainty reporting, three seeds and all registered
comparisons. Do not change the threshold merely to exploit this readout.

Before any deployment claim, the combined policy needs independent scene-level
selection/calibration/confirmation under the approved roles. Those roles remain
closed here. The immediate priority is a correctly referenced risk/utility
target, not a larger Transformer or another generic normalization experiment.

## Verification and Boundaries

144 coordinate-error arrays, 72 causal decision arrays, 396 predecessor metrics,
3,312 metric reductions and 288 independent decompositions pass. The 3,000-draw
bootstrap is recomputed through a separate reduction path. All groups are
resumable and hash-bound; 282 tests across 44 scoped files pass. The full legacy
suite is not run. No new training took place in this diagnosis.

Image-pixel detector tracks, obs8/pred12 rawstride12. Not historical raw-frame
t50, metric, seconds, human gold, physical safety, true3D or foundation evidence.
Historical Stage37 is not recertified. Deployment unchanged, Stage5C off, SMC off.
The research goal remains unfinished.
