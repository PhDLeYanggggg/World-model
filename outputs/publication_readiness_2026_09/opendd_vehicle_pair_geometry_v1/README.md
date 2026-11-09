# OpenDD vehicles: true spacing does not improve matched selection

I tested whether the missing inter-vehicle distances explained the
failure of the [direct pair selector](../opendd_vehicle_pair_learning_v1/README.md).
I compared the correct observed positions with zeroed position channels
and positions shuffled within the same query. All three heads use the
same 6,372 parameters, initial weights, sampled donor pairs, targets and
training budget. The shuffle keeps the position roster and the actor's
own motion, but breaks their alignment. A fourth arm replays the
archived learned additive policy.

The three refits share the same past-defined query and two equal-radius
edits. Spacing uses the [verified observed coordinates](../opendd_vehicle_observed_geometry_v1/README.md)
and causal paths, not future labels or actual collisions. Choices and
separate held-out calibration thresholds were frozen at `74e8525ca`.

| Matched outcome | Archived additive | Motion-only | Permuted | True geometry |
| --- | ---: | ---: | ---: | ---: |
| Recording views | 8,874 | 8,874 | 8,874 | 8,874 |
| Missing easy denominator | 3,730 | 4,163 | 4,233 | 4,339 |
| Pass all original risk checks | 4,379 | 3,822 | 3,750 | 3,612 |
| Also beat CV and OLS4 | 2,975 | 3,059 | 2,964 | 2,890 |

| Direct paired true-geometry contrast | Motion-only | Permuted |
| --- | ---: | ---: |
| Conservative trajectory gain (pp) | -0.001876 [-0.003060, -0.000970] | -0.001882 [-0.002906, -0.000979] |
| Available-step trajectory gain (pp) | +0.000110 [-0.000031, +0.000248] | +0.000031 [-0.000099, +0.000125] |
| Conservative relative-motion gain (native units) | -0.01726 [-0.02222, -0.01248] | -0.01717 [-0.02186, -0.01246] |

Positive contrasts favor true geometry. The direct conservative
trajectory gain is negative versus motion-only at all seven locations.
The available-step contrast is inconclusive; the negative conservative
gain includes the unchanged paired missing-step debit. The
true-minus-motion qualifying count difference is -1.34 per direction
[-2.10, -0.72]. Against the archived additive arm, conservative
trajectory gain is -0.00544 pp [-0.00924, -0.00249]. Intervals use
3,000 paired resamples of seven site means. The 8,874 views repeat
493 recordings over three seeds and six calibration directions.

## Frozen release

| Held outcome | Archived additive | Motion-only | Permuted | True geometry |
| --- | ---: | ---: | ---: | ---: |
| Released recordings (two edits each) | 65 | 98 | 98 | 82 |
| Missing easy denominator | 23 | 36 | 28 | 37 |
| Pass all original risk checks | 32 | 43 | 56 | 35 |
| Also beat CV and OLS4 | 19 | 34 | 44 | 29 |
| Active directions with every release passing both | 0 / 16 | 2 / 24 | 5 / 25 | 2 / 21 |

The release budgets differ. The shuffled negative control has the
largest number of passing releases, not evidence that shuffling is a
useful policy. Sixteen of the true-geometry arm's 29 passes come
from rdb1. None of the policies passes release, and CV remains the
fallback. Keep this negative ablation rather than adding geometry as
a claimed contribution. The priority is now a prospective external
evaluation and a better-supported test of the release rule.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_vehicle_pair_geometry.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_pair_geometry.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_pair_geometry.py tests/test_m3w_songdo_publisher_footprints.py tests/test_m3w_opendd_vehicle_observed_geometry.py tests/test_m3w_vehicle_pair_learning.py tests/test_m3w_vehicle_pair_support.py tests/test_m3w_vehicle_pair_hindsight.py tests/test_m3w_vehicle_support_rank.py tests/test_m3w_vehicle_easy_factor.py tests/test_m3w_vehicle_matched.py tests/test_m3w_vehicle_residual.py
```

`fresh_run`: 189 arm64 Torch fits, 37,800 updates, held scoring.
Four local threads, zero data workers, checkpoints every 100 steps
and PID heartbeats. Twenty-eight focused tests passed.
`cached_verified`: observed geometry, source cohorts, candidate
forecasters and the archived additive comparator. No CREATE job,
external outcomes, manuscript or deployment change.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Per-site readout](readout.json).
