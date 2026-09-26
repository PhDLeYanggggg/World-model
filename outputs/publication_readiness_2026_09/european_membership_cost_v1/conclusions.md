# Membership-Conditional Cost: Completed, Primary Gate Failed

## Material Passport

288 fresh Torch cost heads, 576,000 updates, three seeds and all six source
assignments completed. Frozen membership models, forecasting producers,
reference costs and source arrays are cached_verified. Registration 2c321649
preceded fitting; prediction freeze a5746d3c was pushed before all 36 current
held-locality group readouts. No result-dependent model selection was made.
New policy evaluation and independent confirmation are not_run.

## Main Result

Conditional easy/non-easy harm estimation combined with learned membership
does not pass the preregistered expected-cost gate. These are easy-harm MSE
contrasts on positive forecast disagreement, not trajectory ADE/FDE gains.

| Full-input comparison | Positive / negative / overlapping 95% CIs | Assignment point range, MSE gain % |
|---|---:|---:|
| Conditional versus new direct | 0 / 1 / 5 | -82.52 to +6.90 |
| Conditional versus original mean | 0 / 4 / 2 | -88.84 to +3.81 |
| Conditional versus fixed-probability composition | 6 / 0 / 0 | +51.12 to +81.60 |
| New direct versus original mean | 0 / 5 / 1 | -9.14 to +0.67 |

Learned membership has incremental value within the conditional construction:
replacing it by a fitting-only constant is substantially worse. This is not
enough to beat the established cost model. The constant is a weak composition
control, not the strongest baseline. It retains exactly the same conditional
experts and is not a separately trained predictor.

Tail/all-harm guards also fail. Full top10 harm capture has one negative
interval against original_mean, and all-row H_all MSE has one negative interval
against both direct and original_mean. No policy is promoted.

## Useful Signal and Unresolved Error

Median full conditional harm-event AUROC is 0.56160, versus direct 0.46431
and original 0.48578. Against direct, five AUROC intervals are positive;
against original, three are positive, two negative and one overlaps zero.
Median top10 harm capture is 25.91%, below original 27.40%. Better event ranking
does not establish accurate costs or tail protection.

Conditional cost is worse than original in 53/72 dependent full held views.
Outside-easy rows dominate the excess in 50/53 worsening views, accounting
for 92.90% of the sum of positive excess contributions. These rows have zero
easy-harm target by definition, even when their all-harm target is positive.
The model still assigns them too much easy-harm mass.

Conditional improves fitting MSE over original in only 14/72 views; 13 of
those improvements do not transfer. Held improvement occurs in 19/72. This
is not solely a held-scene overfitting story: the current objective/readout
also underperforms the established fit in most views. Fitting composed
metrics use in-sample membership probabilities and are optimistic diagnostics.

Motion-only does not rescue the result: conditional versus original has zero
positive, one negative and five overlapping MSE intervals. Versus direct it
has one positive, one negative and four overlapping intervals. Thirty of 72
motion-only views have weak positive-event support; full has zero such flags.
Full/motion-only disagreement populations differ, so this is not a matched
population modality ablation.

## Computation and Statistical Limits

Summed new-head fitting time is 459.5835 seconds: direct 225.7003 and
conditional 233.8832. This excludes loading, inference and verification.
Runtime: native arm64 Torch, CPU4 / interop1 / workers0. All checkpoints and
heartbeat logs are local; the 100-update pilot resumed within the fixed budget.
Zero unknown-label training draws. Both new heads have 24,706 parameters;
conditional also uses a frozen 24,641-parameter membership model. Equal new
head budgets are not equal total system cost.

Three seeds are averaged within locality, then four localities per assignment
are resampled 3,000 times. Assignments and windows are dependent; these are
exploratory, non-multiplicity-adjusted source-development intervals, not
independent confirmation. No threshold was refitted. Six selection, twelve
reserved calibration and six confirmation localities stay unused.

## Decision

Retain the previous policy unchanged. Do not promote this cost factorization
or claim new world dynamics. Next, distinguish cost-weighted membership error
from within-easy severity error on source development before another fit;
keep the original readout as the required comparator. See
[failure analysis](failure_analysis.md), [project gap](project_gap.md),
[full results](results.md) and [method positioning](literature_position.md).

Detector-derived pixels and eight observed/twelve predicted annotation steps
only. No metric/seconds, human-gold, physical-safety, true3D, foundation or
submission-ready claim. Stage5C/SMC remain off. The long-term goal is ongoing.
