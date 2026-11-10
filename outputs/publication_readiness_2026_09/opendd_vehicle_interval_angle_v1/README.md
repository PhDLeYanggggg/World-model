# OpenDD: interval supervision finds more options, not a better release rule

The earlier angle selector fitted a worst-case relative-motion gain
as if it were an exact target. I kept its inputs, candidates, network,
optimizer and seed schedule, and penalized the relative-motion estimate
only when it fell outside the paired missing-label interval. The
trajectory and dual-pass losses were unchanged. This is a controlled
application of partial-label regression, not a new interval-loss method;
see [Pukdee, Ke and Gupta](https://proceedings.neurips.cc/paper_files/paper/2025/hash/570641fd4b0340cc8ba6320955de891f-Abstract-Conference.html).

I ran 63 new CPU Torch fits: 21 excluded location pairs, three seeds,
and 200 steps per fit. Predictions for all 126 evaluation directions
were frozen at `63c21e93e` before the new scoring pass. Each threshold
comes from the opposite site. The old direct-control weights reproduced
the same scores and choices, and the old evaluation counts matched
exactly. Of 44,370 repeated donor training pairs, 25,702 intervals
strictly crossed zero and 440 reduced to point labels.

| Fixed original rotations | Selector | Replacements | Qualified over first angle | Strict passes over unrotated path |
| ---: | --- | ---: | ---: | ---: |
| 526 | Direct lower target | 89 | 11 | 55 |
| 526 | Interval target | 145 | 20 | 53 |
| 526 | Direct, calibrated | 0 | 0 | 54 |
| 526 | Interval, calibrated | 2 | 1 | 54 |
| 1,368 | Direct lower target | 160 | 16 | 103 |
| 1,368 | Interval target | 343 | 34 | 99 |
| 1,368 | Direct, calibrated | 5 | 0 | 107 |
| 1,368 | Interval, calibrated | 6 | 1 | 107 |

All arms preserve the original recipient and rotation masks and use
the same angle bank. The number of angle replacements within those
masks is allowed to vary. A qualified replacement improves both
conservative trajectory and relative-motion scores over the first
angle, passes the original 2% checks, and beats both motion references.
Strict passes use the prior comparison against the unrotated path.

At the primary quota, the uncalibrated interval rule finds nine more
qualifying replacements but makes 56 more replacements in total. The
paired conservative trajectory difference is -0.00001474 pp (95%
seven-location bootstrap [-0.00004047, +0.00000419]). For relative
motion, the missing-label interval is [-0.00016284, +0.00018704] in
native units; the observed contribution's bootstrap interval is
[-0.00000223, +0.00002833]. The lower endpoint worsens, but these data
do not establish that the true full-grid effect is negative.

After calibration the trajectory difference is only +0.000000137 pp
(95% interval [0, +0.000000410]). At both quotas the number of strict
passes is unchanged. The interval target made more proposals, but did
not reliably rank which ones were worth releasing. I kept the incumbent.
This is an OpenDD development result, not another independent external
test; no outside outcomes were accessed. The original release rule and
the v190 manuscript remain unchanged.

[Protocol](protocol.md) | [Primary readout](readout.json) |
[Two-sided bounds](relative_bounds.json)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_vehicle_interval_angle.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_interval_angle.py score
.venv-pytorch/bin/python scripts/audit_m3w_vehicle_interval_angle_bounds.py
```

This uses native arm64 Torch, four compute threads and no loader
workers. Fits checkpoint every 100 updates and resume from the same
command; scoring keeps immutable per-direction receipts. The complete
fit/freeze took 108 seconds on the local machine, and the primary
readout took 41 seconds. No CREATE jobs or new training assets were needed.
