# Source-Internal Visual Information: Completed, Negative

## What Was Run

`fresh_run`: 30 real Torch fits, 60,000 optimizer updates, three seeds and five
held physical SDD sites. The 100-update training-only pilot is included in that
budget. Full continuation wall time was 49.62 minutes; summed fitting time was
2,758.09 seconds. All models ran to the fixed final budget, without selection on
held-site scores. This was not a partial run or a NumPy replacement.

Inputs are `cached_verified`: 22,374 complete-label stationary windows, 726
recording-local IDs and 36 videos from the original SDD train40 admission.
Each fit excludes every video and scoped ID of its held physical site. The
6,460 incomplete-label stationary windows remain unscored, not negative.
No main fit example was used for training these models; no sealed role was opened.

The comparison changes only RGB availability. Both arms use the same past
geometry, coverage masks, architecture, initialization, sampled rows and budget.
This tests source-internal information, not a new forecasting policy or official
split. The main native 8-to-12 task and equal-site past-normalized ADE are unchanged.

## Main Result

Positive numbers below would mean a lower Brier score from adding RGB. These are
absolute probability-score differences, not percentage ADE/FDE improvements.

| Held physical site | RGB versus matched mask | RGB versus training constant prior | Positive RGB/mask seeds |
| --- | ---: | ---: | ---: |
| bookstore | -0.021543 | -0.047988 | 0/3 |
| coupa | -0.031010 | -0.039439 | 0/3 |
| deathCircle | -0.028462 | -0.087600 | 0/3 |
| gates | -0.008525 | -0.022617 | 1/3 |
| hyang | -0.010692 | -0.046279 | 0/3 |

The registered equal-site RGB-minus-mask contrast is **-0.020046**, with a
2,000-draw conditional site-block interval **[-0.027587, -0.011995]**. All five
site means are negative; 14/15 seed-site pairs are negative. The equal-site
comparison with the training prior is -0.048785. None of the 15 RGB fits, or the
15 mask fits, beats its own constant training prior on the held site.

Equal-agent weighting does not rescue the aggregate: the equal-site/equal-agent
contrast is -0.016689, conditional interval [-0.026445, -0.006347]. Hyang has a
small positive agent-weighted mean (+0.000773), with an interval crossing zero;
this exception is retained rather than hidden. Per-video and per-seed tables
are available in `video_metrics.csv` and `fit_metrics.csv`.

Five exposed physical sites and overlapping training folds are not independent
confirmation. The bootstrap is a conditional stability description, not a
population-wide significance certificate or formal risk guarantee.

## Failure Taxonomy

| Explanation | Evidence | What remains unproved |
| --- | --- | --- |
| Only source-to-main domain gap | Insufficient explanation: the matched RGB arm also fails between SDD source sites | Does not isolate camera, behavior and label differences inside SDD |
| No learning / broken runtime | Not supported: all updates finish; RGB mean training Brier falls to 0.202032 versus mask 0.211919, but held Brier rises to 0.297592 versus 0.277546 | Finite budget does not establish convergence or optimal capacity |
| Fit-specific visual patterns | Consistent with better training fit and worse held-site scores | The experiment does not causally identify background shortcut learning |
| Prior / calibration shift | Held positive rates range 25.27%-59.91%; both arms lose to their own training priors in all 30 fits | Calibration might help, but no held-label recalibration is performed |
| Useful sample-specific probability variation | Not established: RGB's varying-prediction Brier term versus a constant prior is negative in all five sites | Weak ranking information may remain; this is not proof of zero mutual information |
| Incorrectly interpreted supervision | The positive label is any annotation-center change; its median maximum displacement is only 2.06 annotation pixels | Small motion is not automatically noise or a wrong label |
| Missing past rows | Not the observed failure: all 22,374 histories have exact eight-frame/agent joins, with original past crops | Past RGB availability does not prove visible body state or intention |

## Label Frequency Is Not Forecasting Cost

Of 10,039 positive labels, 8,583 (85.50%) move less than 0.1 of the current box
diagonal. Only 244 windows reach half a box diagonal. All histories contain at
least one generated annotation row; 99.686% of sampled past rows carry that flag.
This remains the approved offline annotated-history task, not strict sensor-as-of
prediction, and these labels are not human motion-intention gold.

A supplementary source-only cost census adds an important qualification. The
244 larger-motion windows are just 1.091% of stationary windows but contribute
26.155% of stationary CV ADE error mass. Conversely, the 8,583 small-change
windows still contribute 43.406% of that error mass. Therefore neither counting
all positive labels equally nor discarding all small changes is justified as a
substitute for the forecasting objective. These are window-pooled source error
contributions, not main-task results, neural gains or new selection thresholds.
The descriptive current-box denominator is distinct from the older median-past-
box definition, even though their half-box counts happen to agree here.

## Verification and Claim Boundary

Thirty checkpoint prediction replays are exact. Fifteen paired training-stream
and normalization checks pass. Completed resume preserves 91 immutable artifacts
plus the report hash and adds zero fits or updates. All 46 focused tests pass;
the full legacy repository suite is not represented as rerun. Scientific figure
rendering was visually inspected. Private data, predictions and weights remain
local; only aggregate evidence, code and configs are committed.

Registration SHA: `a2038307eab36cc62ae783eedf3f359747965f2ddcc17cf4f661d83273fab74d`.
Report SHA: `4a515cd3b56d1f29bb9e30e7349682bce030f174bec397fda265d4a70aae1142`.
Registration was pushed before training at `a91e5795`.

No model promotion, new trajectory improvement, joint-intervention advantage,
independent confirmation or submission readiness is established. SDD remains
pixel-space; this source task is +144 raw frames, not physically time-equated to
the main task. No metric/seconds, true-3D or foundation claim. Stage5C and SMC
remain disabled. Historical exploratory scores are not reinstated as clean tests.

## Next Concrete Action

The next controlled repair should address supervision-to-forecast-cost alignment
and visible event support before another source-to-main alignment or larger-CNN
run. Preserve the existing population, report small- and larger-change costs, and
test a single training-objective change against the same mask/RGB controls before
claiming forecasting utility. Do not select a favorable scene or threshold from
this matrix. Any change to the formal primary metric, split or observation
contract requires a separate explicit decision. The overall research goal remains
active and unmet; this completed diagnostic is one evidence step, not its endpoint.
