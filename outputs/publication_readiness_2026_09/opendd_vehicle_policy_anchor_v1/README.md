# OpenDD: train on the recipient policy, then check the nominee bottleneck

I held the training query, path families and peak displacement fixed,
and changed only which two recipients were chosen. The policy arm uses
the original additive merit conditional on those families. It is
not an unrestricted new candidate bank. Its nested chooser excludes
the training example's location and both outer locations; its
inner forecasts also exclude the location used to train that
chooser. Both arms use one sample per recording in each fold.

Among 22,185 matched donor samples, the policy changes the set of
actors in **21,141** and the nominated actor/angle in **18,178**.
The exact label is positive in 930 policy samples and 876 random
samples. The two arms share 42,962 learned-family recipient slots
out of 44,370. Neither the held nominee nor the budget changes.

## Matched results on the frozen proposals

| Screen | Strict / 526 | Risk / 526 | Strict / 1,368 | Nominee AUROC |
| --- | ---: | ---: | ---: | ---: |
| Original action margin | 54 | 261 | 107 | 0.516 |
| Random recipients, pointwise | 36 | 284 | 83 | 0.518 |
| Policy recipients, pointwise | 46 | 277 | 100 | 0.583 |
| Random recipients, budget rank | 36 | 284 | 88 | 0.545 |
| Policy recipients, budget rank | 35 | 277 | 89 | 0.577 |

At 526 rotations, the pointwise policy-minus-random conservative
trajectory contrast is -0.000012 pp [-0.000058, +0.000023].
For budget ranking it is -0.000016 pp [-0.000039, +0.000007].
Relative motion is worse in both comparisons, respectively
-0.000409 native units [-0.000699, -0.000159] and
-0.000422 [-0.000709, -0.000176]. At the 15% quota, the
pointwise arm's strict pass rate rises by +0.218 pp
[+0.008, +0.446], while trajectory utility falls by -0.000125 pp
[-0.000340, -0.000007]. More passes do not imply more utility.

| Separately calibrated screen | Rotations | Risk passes | Risk, both references and positive incremental gain |
| --- | ---: | ---: | ---: |
| Original margin | 94 | 37 | 11 |
| Random recipients, pointwise | 35 | 20 | 4 |
| Policy recipients, pointwise | 23 | 11 | 5 |
| Random recipients, budget rank | 23 | 15 | 6 |
| Policy recipients, budget rank | 14 | 12 | 5 |

Neither learned screen replaces the original. Intervals use 3,000
paired location draws after averaging directions and seeds. These
8,874 views repeat 493 OpenDD development recordings. No new
external outcomes were opened.

## The shortlist has useful room that a screen cannot recover

The [shortlist audit](topk_capacity.json) scores only the old ego
head's frozen order. It changes no trained model or threshold.

| Angles retained | Views with a strict witness | Hindsight cap at the 526 budget | Hindsight cap at the 15% budget |
| --- | ---: | ---: | ---: |
| 1 | 420 | 238 | 420 |
| 3 | 802 | 329 | 781 |
| 6 | 916 | 345 | 865 |
| All 24 | 1,393 | 388 | 1,078 |

The first three candidates nearly double strict support relative
to top one, and increase it in all seven locations. At each
budget the cap honors the original count in each direction.
This is outcome-aware capacity, not a new trained policy. It
motivates comparing learned choice within a three-candidate
shortlist to fixed top one at the same intervention count.
For a later independent vehicle check, the
[rounD publisher](https://levelxdata.com/round-dataset/) lists
three drone-recorded locations and academic access. Its source
overlap and causal input provenance still need verification.
No request or data download was made.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/fit_m3w_vehicle_nested_additive.py
.venv-pytorch/bin/python scripts/run_m3w_vehicle_policy_anchor.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_policy_anchor.py score
.venv-pytorch/bin/python scripts/audit_m3w_vehicle_nominee_topk.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_policy_anchor.py
```

`fresh_run`: 105 additive-chooser fits and 252 screen fits,
71,400 total arm64 Torch updates across three seeds, and 126
paired direction evaluations. The pre-score freeze was pushed
as `5deff06bb`. Checkpoints every 100 steps preserve optimizer
and RNG state; the runtime logs the PID. The targeted regression
suite passed 150 tests. `cached_verified`: earlier forecasts,
angle teachers, proposal bank, calibration roles and original
margin. No new external release or Stage5C/SMC.

[Protocol](protocol.md) | [Teacher freeze](additive_freeze.json) |
[Pre-score freeze](freeze.json) | [Readout](readout.json).
