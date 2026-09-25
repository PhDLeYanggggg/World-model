# More Supported Pairs Did Not Establish Safe Neural Superiority

## Result Passport

- Fresh work: 36 Torch risk heads, 72,000 updates, three seeds and 216 evaluated views.
- Reused with hashes: forecasts, utility heads, preceding ranked controls and source-excluded producer chains.
- Not run: new trajectory fitting, independent calibration/final confirmation or remote training.
- Decision: no deployment promotion; no stable safe neural advantage; not submission ready.

## What Changed and What Worked

We removed undefined event-mass rows before forming within-locality ranking
pairs. The same rows still participate in every original loss. Capacity,
initialization, minibatch draws, ranking formula, loss coefficient, training
steps, forecasts and the 2% predicted-risk rule were unchanged. The complete
protocol was committed before fitting as fb876710.

Neural/easy valid pair draws increased from 167,291 to 541,866, about 3.24 times.
Damping/easy increased from 104,799 to 309,201. These are repeated training
exposures, not independent observations. All nine neural/easy fixed fitting
batches finish with lower total loss, and eight with lower ranking loss.
This fixes a supervision-efficiency issue but does not prove generalization.
See [fitting diagnostics](fitting_diagnostics.md) and [fixed losses](training_loss.svg).

Neural/all pair counts, full saved score banks and all nine full-policy results
are exactly unchanged. That is a useful negative control: its labels were
already supported, so the repair should not alter its behavior. Damping/all
has four more pairs across 18,000 updates and only tiny policy differences.

## Controlled Evaluation

Each cell counts positive / negative conditional 95% intervals among nine
fold-seed comparisons. Intervals containing zero are not counted in either
direction. Components use percentage points of CV-normalized all-ADE gain.

| Candidate / event | Full-policy change | Ordering at old counts | Ordering at new counts |
|---|---:|---:|---:|
| Neural / all | 0 / 0, exactly unchanged | 0 / 0 | 0 / 0 |
| Neural / easy | 2 / 1 | 1 / 1 | 2 / 1 |
| Damping / all | 0 / 0 | 0 / 0 | 0 / 0 |
| Damping / easy | 0 / 7 | 0 / 6 | 0 / 6 |

Neural/easy full-policy changes range from -0.0616 to +0.1588 pp, with mixed
matched-count evidence. Damping/easy worsens in all nine point comparisons,
from -2.8608 to -0.0345 pp. It is not valid to present a smaller neural-to-damping
gap as stronger neural dynamics when the damping controller itself worsened.
[Every registered view](results.md) and the [all-contrast figure](ranking_comparison.svg)
retain both count anchors and full/common support effects.

## Strong Control and Preservation

Against equally protected damping, all 18 neural all-ADE point comparisons
remain negative; 17 intervals are strictly negative. Relative differences range
from -2.1912% to -0.1995%. Hard-subset comparisons have no positive interval and
16 negative intervals; the lone positive point is only +0.0309%, not a supported
hard-case improvement. Some easy-subset comparisons favor neural.

| Arm | Observed preservation | Positive-easy limit | Zero-CV harm |
|---|---:|---|---:|
| Frozen ranked neural | 6/18 | 18/18 within 2% | 12 views |
| Supported-pair neural | 7/18 | 18/18 within 2%; worst 0.0606% | 11 views |
| Frozen ranked damping | 15/18 | Three failures | No observed harm |
| Supported-pair damping | 16/18 | Two failures; worst 2.4671% | No observed harm |

Six neural views have no zero-CV examples, not established protection. The new
seventh preserved view is only one observed improvement, not a general solution
for unsupported zero-reference cases. Deployment remains unchanged.

## What This Rules Out and What Remains

More available pairs and lower fitting losses were not sufficient to repair
excluded-locality ordering. Pair scarcity alone is therefore not a sufficient
explanation under the tested budget. This does not prove additional training,
representation changes or better source support could never help.

A separate [constructed counterexample](estimand_counterexample.md) shows a
structural limitation: ranking realized H/(B+H) can oppose ranking conditional
E[H|x]/E[B|x]. With the actual auxiliary, underestimating an unsafe state's harm
reduces its ranking loss in that example. It does not prove this caused the
real-data failure or describe the optimum of the full combined training loss.

The next controlled change should target this estimand mismatch, not add more
pair variants or loosen deployment thresholds. A cross-moment pair target has
the desired conditional sign under independence, but dependence, random batch
normalization and weight variance still require testing. It has not been trained
here. Zero-reference support remains a separate unresolved problem.

## Verification and Boundaries

All 36 checkpoint replays, sampled feature/preprocessing checks and draw counts
match their bound records. All 216 views reproduce and 36 old controls match
exactly. Separate scalar sorting, coordinate arithmetic and bootstrap reductions
verify 216 decisions, 864 metric reductions and 108 decompositions. There are
247 passing tests across 39 scoped files, not a full legacy-suite run.
All required phases have finished. [Receipt](completion_checks.json),
[execution notes](execution_notes.md), [gates](gates.md), [model card](model_card.md).

Each fit uses four fitting/eight complete-chain-excluded localities. All twelve
European Squares localities are opened development sources. Three seeds and
3,000 locality-bootstrap resamples give dependent, conditional, unadjusted
intervals, not independent confirmation. Detector-track image pixels,
obs8/pred12 rawstride12; not historical t50, seconds, metric, human gold,
physical safety, true 3D or foundation evidence. Historical Stage37 is not
recertified. Reserved roles stay closed. Stage5C and SMC stay off.
