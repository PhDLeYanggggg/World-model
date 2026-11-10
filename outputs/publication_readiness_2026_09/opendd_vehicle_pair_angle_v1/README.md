# OpenDD vehicles: supervise the replacement directly

I compared two equal-capacity models on the same incumbent and alternative
angle pairs. Both use identical past-only inputs, risk/reference supervision,
initialisation, draws and optimizer. The first learns the difference of two
lower gains against the unrotated path. The second learns the conservative
gain of switching directly from the incumbent. On a missing target, the
direct comparison charges the maximum change between the two proposed paths
once. The difference between separate lower bounds is not itself a lower
bound on the replacement.

The 126 native Torch fits cover 21 excluded site pairs, three seeds and two
targets at 200 updates per fit. Donor upstream models exclude the donor,
calibration and held sites. I froze the choices and opposite-site thresholds
at `e211c486d`, before the 126 held directions were scored. The first-angle,
pointwise-utility and prediction-agreement comparisons reproduced exactly.
There was no new external scoring.

## Label change and matched result

Across 44,370 training pairs (repeated across folds and seeds), the positive
trajectory label differs in 2,087 (4.70%) and the positive relative-motion
label in 8,412 (18.96%). The two positive labels coincide in 13,393 pairs with
the anchor-difference targets and in 7,062 with the direct targets.

| Original mask | Rule | Angle replacements | Qualifying replacements | Relative negative lower bound | Strict passes over unrotated path |
| ---: | --- | ---: | ---: | ---: | ---: |
| 526 | First angle | 0 | 0 | 0 | 54 |
| 526 | Pointwise agreement | 96 | 9 | 79 | 52 |
| 526 | Matched anchor-difference | 209 | 22 | 163 | 46 |
| 526 | Matched direct | 89 | 11 | 70 | 55 |
| 526 | Calibrated anchor-difference | 3 | 0 | 3 | 54 |
| 526 | Calibrated direct | 0 | 0 | 0 | 54 |
| 1,368 | First angle | 0 | 0 | 0 | 107 |
| 1,368 | Pointwise agreement | 309 | 30 | 221 | 106 |
| 1,368 | Matched anchor-difference | 572 | 39 | 440 | 90 |
| 1,368 | Matched direct | 160 | 16 | 119 | 103 |
| 1,368 | Calibrated anchor-difference | 4 | 0 | 4 | 106 |
| 1,368 | Calibrated direct | 5 | 0 | 4 | 107 |

A qualifying replacement has positive direct conservative trajectory
and relative-motion gains over the original first angle and passes the
original 2% risk and both reference checks. The strict column is the
previous criterion against the unrotated path, not the same count. The
original masks are unchanged, so an unchanged angle still counts as one
original rotation. There are 66 and 168 qualifying views in hindsight,
respectively. Fewer replacements did not yield more qualifying changes.

## Missing targets and paired uncertainty

| Direct minus matched anchor-difference | At 526 | At 1,368 |
| --- | ---: | ---: |
| Conservative trajectory gain (pp) | -0.00000129 | -0.00008360 |
| 95% bootstrap of that lower gain (pp) | [-0.00001213, +0.00000922] | [-0.00019505, -0.00000422] |
| Missing-label relative-motion bounds | [-0.00024653, +0.00021565] | [-0.00107274, +0.00078508] |
| Observed contribution to full-grid relative gain | -0.00001544 | -0.00014383 |
| 95% bootstrap of observed contribution | [-0.00004065, +0.00000882] | [-0.00037373, +0.00002440] |
| Changed pair-steps with observed labels | 19,131 / 22,428 | 47,067 / 56,136 |

Positive means lower error. Relative motion stays in the dataset's native
units. The observed contribution uses the full-grid denominator, not a
renormalised complete-case ADE. The missing-label interval describes
possible full-grid values; it is not a sampling interval. Its lower
endpoint reproduces the original scoring at every site. Both intervals
span zero, and the separate visible-data contributions are inconclusive.
A negative lower bound blocks a claim of conservative improvement;
it does not prove that the full-grid effect was harmful. The
[two-sided audit](relative_bounds.json) and [primary readout](readout.json)
give all point estimates and 3,000 paired location bootstrap draws
summarised by their endpoints. The 493 recordings and 8,874 repeated
views are not 8,874 independent samples.

**Decision:** keep the original angle policy. This ablation corrects a
training target, but neither budget yields a calibrated improvement.
This is a development result, not new confirmation. A more informative
next test than another threshold sweep is a prospectively frozen test
on an acquisition source with dense pair labels. That separates lack
of predictable relative benefit from lack of identifiable benefit.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_vehicle_pair_angle.py freeze
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_vehicle_pair_angle.py score
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_vehicle_pair_angle_bounds.py
```

The first command resumes checkpoints saved every 100 updates and reuses
complete packet receipts. It needs the previous local, source-excluded
weights. The second reuses scored directions; the third is a post-score
diagnostic on those frozen choices and changes no threshold.
