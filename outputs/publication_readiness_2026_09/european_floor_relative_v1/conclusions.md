# Reference Alignment Alone Does Not Repair the Controller

Completed on 2026-09-25. Fresh training and fresh evaluation; frozen predecessor
forecasts and the indexed population are cached_verified. Registration `1049876b`
was pushed before fitting. Both complete decision banks preceded either new
evaluation. No threshold, seed, normalizer or deployment winner was selected.

## What Was Learned

The actual protected fallback should be included in the comparison. However,
changing both learned cost targets from CV-relative to floor-relative does not
improve the matched controller. It loses on 33/36 overall ADE point estimates;
20 conditional paired intervals favor the matched CV-target control and none
favor the floor-target arm. All 18 all-event comparisons lose. The three positive
points are in easy-event views, none with a positive interval. This rejects a
simple target-reference repair under the registered setup, not the general
possibility of useful relative-risk learning.

| Mode / event | Both floor targets: all-ADE gain over protected floor | Gain over matched CV targets | Positive / negative matched intervals |
|---|---:|---:|---:|
| Batch / all | 0.2651% to 1.6409% | -0.4413% to -0.0967% | 0 / 7 |
| Batch / easy | 0.1270% to 0.2979% | -0.0675% to 0.0021% | 0 / 2 |
| Fitting / all | 0.2650% to 1.6254% | -0.5394% to -0.0569% | 0 / 8 |
| Fitting / easy | 0.1225% to 0.3187% | -0.0650% to 0.0065% | 0 / 3 |

Each cell contains nine fold/seed views. Intervals are paired locality-bootstrap
intervals, conditional on opened development and fitted models, not simultaneous
or independent confirmation. The gain denominator is the named reference for
each column; these columns are not differences in CV-normalized percentage points.

Changing only utility targets has 2 positive and 21 negative matched all-ADE
intervals across 36 views. Changing only risk targets has 0 positive and 13 negative.
Neither establishes a consistent repair. All arms and the old rebased control
remain in `results.md` and the per-fold tables, including unknown-label switches.

There is still a useful positive finding: the both-floor policy has positive
all-ADE, complete-label ADE and endpoint FDE intervals against the actual floor
in all 36 views. Hard-ADE points are all positive, with 33 positive intervals.
Those facts do not prove that floor-target training caused the gain: the matched
CV-target controller is usually better, and the old rebased policy also has
positive overall intervals. No new trajectory forecaster was trained.

## What Still Fails

All 144 newly fitted-policy views stay below 2% worst-locality positive-easy
degradation; the largest is 0.3001%. The both-floor arm improves positive-easy
error in every evaluated locality. Nevertheless, every arm harms zero-CV cases
in 24/36 views. The other 12 views have no zero-CV examples; they do not certify
zero-reference preservation. These are repeated views of only four underlying
rows, one locality, two future labels per row and no endpoint.

The post-hoc causal-history check is specific: all four rows have zero last-step
speed but nonzero total past path length, so the existing past-motion guard
admits them. Both folds evaluating those rows have zero zero-CV fitting examples.
This is a support and stop/start-guard gap. It is not permission to infer future
stationarity, fit a rule to those outcomes or claim full-horizon safety.

Average hard improvement also hides local degradation: the both-floor arm's
worst hard locality loses 2.8055% in batch mode and 1.8413% in fitting mode against
the protected floor. The two-source fitting-floor to four-source evaluation-floor
producer shift remains unresolved. Coverage changes and ordering changes are not
separated by this experiment, and loss decrease does not settle either question.

## Evidence and Decision

234 new Torch heads, 468,000 updates, 119,808,000 repeated training draws. 90 heads
produce honest inner floors; 144 are matched main heads. All 234 checkpoints replay
their recorded inference prefix exactly (up to 4,096 rows), and all score arrays
are hash-bound. No unknown-label row was sampled for supervised training.
Risk fixed-training loss decreases in 129/144 heads. This is optimization evidence,
not a validation score. Utility traces use changing minibatches.

Verification passes:144 causal decision arrays,144 coordinate-error arrays,
2,160 separate metric reductions and180 exact preceding-control comparisons.
301 scoped tests in 47 files pass; the unrelated full legacy suite was not run.
The population remains 318,969 overlapping windows from 12 opened localities:
193,705 complete,118,217 partial and7,047 unknown future labels.

Deployment stays unchanged. The next focused test should isolate support-aware
stop/start abstention and selected-harm calibration, with a matched-coverage
control, before spending another round on loss variants. It must be registered
before its own readout and retain all relevant negatives. Reserved selection,
calibration and confirmation roles remain closed.

These are image-pixel obs8/pred12 rawstride12 results on released detector tracks,
not verified human gold, metric/seconds, historical t+50 recertification, physical
safety, true3D, foundation success or submission readiness. Stage5C and SMC remain
off. The overall research goal is not complete.
