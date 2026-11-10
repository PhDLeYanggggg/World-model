# OpenDD vehicles: require agreement before changing the angle

I kept the same recording masks, recipients and three-angle menus. I
reused the 63 source-excluded utility heads and their original angle
choosers. The new rule replaces the first angle only if its predictions
favor both trajectory and relative-motion utility, and its predicted
original-risk pass probability is no worse. A separate arm calibrates
the replacement threshold on the opposite location. No new model was
fitted. The 126 directional choices and thresholds were frozen before
held scoring in `df7f6d5dc`. Both previous comparators reproduced,
including the paired effects at each of the seven locations.

## Same recordings, different angle choices

All counts are repeated recording views, not unique recordings.
The 526 and 1,368 denominators are identical across arms.

| Rotations | Angle rule | Replacements | Qualifying joint improvements | Relative harm | Strict passes over anchor |
| ---: | --- | ---: | ---: | ---: | ---: |
| 526 | First angle | 0 | 0 | 0 | 54 |
| 526 | Prior utility | 380 | 39 | 299 | 56 |
| 526 | Trajectory only | 350 | 37 | 274 | 56 |
| 526 | Agreement | 96 | 9 | 79 | 52 |
| 526 | Calibrated agreement | 1 | 0 | 1 | 54 |
| 1,368 | First angle | 0 | 0 | 0 | 107 |
| 1,368 | Prior utility | 873 | 93 | 627 | 119 |
| 1,368 | Trajectory only | 858 | 88 | 623 | 114 |
| 1,368 | Agreement | 309 | 30 | 221 | 106 |
| 1,368 | Calibrated agreement | 23 | 3 | 16 | 105 |

Qualifying joint improvement requires both direct conservative gains
against the first angle, plus the original risk and two reference
checks. Strict passes retain the previous definition against the
unrotated anchor. These answer different questions.

| Hindsight: at least one alternative | At 526 | At 1,368 |
| --- | ---: | ---: |
| Trajectory improves | 323 | 877 |
| Relative motion improves | 145 | 435 |
| Same angle improves both | 132 | 411 |
| Same angle improves both, with risk and both references | 66 | 168 |

The exact paired bounds charge each missing target once; they do not
subtract two independent lower bounds. The last row is not an
achievable causal policy score.

## Paired effects against the original first angle

| Rotations | Angle rule | Conservative trajectory gain (pp) | Relative-motion gain (native units) |
| ---: | --- | ---: | ---: |
| 526 | Agreement | -0.00000797 [-0.00002247, +0.00000110] | -0.00004493 [-0.00007957, -0.00001500] |
| 526 | Calibrated agreement | -0.00000040 [-0.00000119, 0] | -0.00000109 [-0.00000327, 0] |
| 1,368 | Agreement | +0.00007658 [+0.00001902, +0.00015524] | -0.00023279 [-0.00037578, -0.00010713] |
| 1,368 | Calibrated agreement | +0.00002846 [-0.00000192, +0.00008680] | -0.00003112 [-0.00007806, -0.00000481] |

Positive means better. The [readout](readout.json) includes all arms,
observable FDE, site means, and harm counts. Intervals use 3,000 paired
location draws, after averaging directional results and seeds, over
the same 493 recordings (8,874 repeated views before masks). No
independent confirmation data was opened.

The agreement rule reduces the number of harmful replacements mostly
by making fewer of them. At 526 rotations the fraction with relative
harm is 79/96 versus 299/380 for the prior utility rule. Of 66 hindsight
opportunities, it realizes nine, versus 39 for the prior utility rule.
Only 6 and 14 of 126 calibration directions have nonempty thresholds,
respectively. Neither calibrated arm passes; deployment stays unchanged.
The failure points to what the heads learn: two independently predicted
effects against the unrotated anchor do not reliably estimate an
incremental change between nominees. A direct pairwise target against
the original angle is the next controlled comparison, if another
development fit is run. A stronger main claim still needs independent
evidence beyond OpenDD.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_vehicle_angle_agreement.py freeze
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_vehicle_angle_agreement.py score
```

Both commands resume by reusing completed direction receipts. The
first uses the original local forecast packets and checkpoints; the
second uses the committed freeze and only the same OpenDD outcomes.
