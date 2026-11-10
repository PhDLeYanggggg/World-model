# OpenDD: learn the outcome of the nominated action

The opportunity score could find recordings with a useful angle but
could not reliably select where to apply the actual nominated angle.
Here the target is the exact nominee: did it pass the unchanged
risk checks and both motion references, preserve conservative
trajectory utility, and improve relative motion?

Training nominees come from heads that exclude the example's
source and both outer locations. Their own training uses forecasts
that also exclude each inner training source. On the outer
locations, the actor, angle and displacement stay exactly as in
the previous experiment. I compared equally sized pointwise and
budget-ranked heads on the same examples. The latter weights the
highest-scoring negatives within each source; the motivation is
related to [partial-AUC optimization](https://proceedings.mlr.press/v28/narasimhan13.html),
but the loss is a logistic surrogate, not that structural SVM.

## Same action, same number of rotations

| Screen | Strict / 526 | Risk / 526 | Risk and both references, all views | Strict / 1,368 | Nominee AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Action margin | 54 | 261 | 3,001 | 107 | 0.516 |
| Opportunity score | 41 | 247 | 2,966 | 102 | 0.664 |
| Exact nominee, pointwise | 41 | 279 | 2,963 | 105 | 0.605 |
| Exact nominee, budget rank | 40 | 277 | 2,975 | 96 | 0.549 |

At the 526 budget, the budget ranker is below margin by
**0.000116 pp** conservative trajectory utility
(95% location interval [-0.000233, -0.000029]) and
**0.000731** native units of conservative relative gain
[-0.001249, -0.000279]. The trajectory difference is negative
at six of seven locations; the worst is -0.000420 pp.
Against pointwise training, budget ranking changes conservative
trajectory utility by +0.000004 pp [-0.000009, +0.000020]
and worsens the relative comparison. More risk passes did not
translate into more successes under the full criterion.

| Separately calibrated screen | Rotations | Original risk passes | Risk, both references and positive incremental gain |
| --- | ---: | ---: | ---: |
| Action margin | 94 | 37 | 11 |
| Opportunity score | 29 | 11 | 2 |
| Exact nominee, pointwise | 28 | 14 | 1 |
| Exact nominee, budget rank | 25 | 17 | 7 |

There is no new release. Keep the previous policy. OpenDD here is
development evidence, not an independent external result. No new
external outcomes were opened. Counts come from 8,874 repeated
views of 493 recordings; intervals average seed/direction scores
and resample seven locations, not individual views.

## Next test

Only 1,449 of 66,555 repeated donor nominees (2.18%) were strict
positives, against 420 of 8,874 (4.73%) on the fixed outer menu.
The [input-only audit](anchor_shift.json) isolates one reason to
test the anchor generator. At the same 17,748 recipient slots,
**98.34%** of policy-selected paths are from the learned forecast
family, versus **34.60%** expected under uniform sampling of
legal families. The two actors have the same family in **96.92%**
of policy-selected views, versus **33.34%** expected under that
sampling. The audit splits 27 tied family matches evenly, loads
no future arrays, and changes no choices. It identifies a feature
distribution difference, not a causal explanation of the failed fit.

Next, match the donor path-family distribution while keeping the
loss, geometry and held proposals fixed. If matching families is
insufficient, test donor anchors selected by a source-excluded
copy of the actual incumbent instead of changing the loss again.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/fit_m3w_vehicle_nested_nominees.py
.venv-pytorch/bin/python scripts/run_m3w_vehicle_nominee_rank.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_nominee_rank.py score
.venv-pytorch/bin/python scripts/audit_m3w_vehicle_nominee_anchor_shift.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_nominee_rank.py
```

`fresh_run`: 105 forecast fits, 105 inner nominee fits and
126 screen fits on native arm64 Torch, for 88,200 total updates.
All 210 teacher checkpoint hashes and source complements were
verified. Checkpoints every 100 steps, single process and PID
heartbeats; existing matching receipts resume without refitting.
The choices and thresholds were pushed as `1869e9bb6` before
held scoring. `cached_verified`: original proposals, cohort
and the outer nominee weights. The final targeted regression
suite passed 147 tests.

[Protocol](protocol.md) | [Teachers](teacher_freeze.json) |
[Pre-score freeze](freeze.json) | [Readout](readout.json).
