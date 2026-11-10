# OpenDD: finding a possible edit is not the same as using one

I froze one proposed angle per recording using the previous ego
action-value head, including recordings where that head would
normally abstain. The new recording score only decides where to use
that angle. Query, recipients, per-step displacement, candidate,
and risk rules are identical across the three screens.

## What the screen learned

| Score | AUROC: any qualifying rotation | AP: any rotation | AUROC: nominated rotation qualifies |
| --- | ---: | ---: | ---: |
| Ego action margin | 0.5531 | 0.2301 | 0.5155 |
| Learned opportunity | 0.6415 | 0.2771 | 0.6637 |
| Fixed probability-margin product | 0.5880 | 0.2656 | 0.5395 |

Paired location-bootstrap AUROC changes for the opportunity
screen versus margin: +0.0884 [0.0597, 0.1175] for any rotation,
and +0.1482 [0.0679, 0.2593] for the exact nominee. The two AUC
contrasts use the same 3,000 draws as the performance contrasts:
subtract the two stored location means before resampling.

## What it delivered at the same number of rotations

| Screen | Strict successes / 526 | Strict successes / 1,368 | Risk and both references, all 8,874 at the 526 budget |
| --- | ---: | ---: | ---: |
| Ego action margin | 54 | 107 | 3,001 |
| Learned opportunity | 41 | 102 | 2,966 |
| Fixed probability-margin product | 46 | 104 | 2,980 |
| Keep original directions | No rotations | No rotations | 2,975 |

The 526 budget reproduces the original ego choices exactly, direction
by direction. The 1,368 budget is 15% rounded up per direction.
At 526 rotations the opportunity screen loses **0.000111 pp**
conservative trajectory utility against margin (95% location interval
[-0.000212, -0.000036]) and **0.000600** native units of
conservative relative gain [-0.001030, -0.000227]. Its conservative
trajectory contrast at the 15% budget is -0.000010 pp
[-0.000109, +0.000073]. Overall ranking gets better without
making the relevant operating point better.

The exact menu can improve 1,393 views. The nominee qualifies
in only **420** even before screening. With the 526 quota and the
same number per evaluation direction, the perfect hindsight screen
could capture at most **238** (sum of the smaller of the direction's
quota and successful nominee count). At 15% the corresponding cap
is 420. These counts identify two limits: the existing nominee misses
most legal opportunities, and the actual screen also misses many
viable nominees. Neither ceiling is a deployable result.

## Frozen calibration

| Screen | Held rotations | Original risk passes among those | Risk, both references and positive incremental gain |
| --- | ---: | ---: | ---: |
| Margin | 94 | 37 | 11 |
| Opportunity | 29 | 11 | 2 |
| Product | 15 | 7 | 3 |

This is the original 2% rule and separate calibration source, not
an assumption that unchanged views pass. No arm supports promotion.
There were 25, 20 and 18 nonempty calibration prefixes across the
126 directions, respectively. Held counts are not an independent
external validation. No external outcomes were opened.

The next training target should be **success of the exact proposed
action at the operating budget**, with the candidate cross-fitted
out of its own source. Keep the current margin and a stronger
nominee as separate controls rather than treating a better AUC
as a deployment criterion.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_vehicle_opportunity.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_opportunity.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_opportunity.py
```

`fresh_run`: 63 arm64 Torch fits over three seeds, 12,600 updates,
66,555 repeated donor examples (8,834 positive), and 126 frozen
direction scores. `cached_verified`: source-excluded donor forecasts,
ego action head, geometry and original recipient rosters.
Checkpoints every 100 steps and PID heartbeats support resume;
the immutable freeze was pushed as `a9fb0fbf3` before scoring.
The regression run passed 144 targeted tests. Uncertainty resamples
seven locations, not 8,874 independent observations; there are
493 distinct recordings.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Grouped scores and intervals](readout.json).
