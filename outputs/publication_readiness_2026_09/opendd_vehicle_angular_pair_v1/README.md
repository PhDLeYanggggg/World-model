# Does coordinating angles improve two fixed vehicle forecasts?

The sparse OpenDD pedestrian controller has only ten query-seed
views where both actors are edited. I moved this comparison to the
existing two-edit vehicle development roster: 8,874 repeated views
from 493 distinct recordings. Both additive-selected actors and
every step's CV-relative displacement norm stay fixed. There are
17,748 edits per arm. No new data split or external score is used.

| Exact finite-menu question | Views / 8,874 |
| --- | ---: |
| Original additive passes risk and both references | 2,975 |
| Any pair of angles passes both checks | 3,802 |
| Also no conservative trajectory loss and better relative motion | 1,431 |
| The original additive already passes and such a relative win exists | 1,021 |
| A shared angle can achieve a relative win | 380 |
| Rotating just one vehicle can achieve a relative win | 1,393 |
| Rotating both can achieve a relative win | 869 |

The last three rows overlap. Only 38 of the 1,431 views require
two rotations within this menu. The witnesses preserve the
original 2% rule, are outcome-aware, and do not show that a
past-only policy can identify the direction.

| Past-only rule | Dual passes | Risk passes | Undefined |
| --- | ---: | ---: | ---: |
| Additive, no rotations | 2,975 | 4,379 | 3,730 |
| Shared nonzero angle | 565 | 2,031 | 3,730 |
| Independent nonzero angles | 544 | 1,879 | 3,730 |
| Coupled nonzero angles | 568 | 2,090 | 3,730 |

Coupling adds a relative-trajectory cost against OLS4 and leaves
both rotations forced. Compared directly with independent angles,
the conservative trajectory difference is -0.00568 pp
(95% location interval [-0.01008, -0.00208]); the conservative
relative-motion difference is -0.04458 native units
[-0.07098, -0.02273]. A few more threshold passes do not mean
greater utility. The [one-rotation comparison](../opendd_vehicle_single_rotation_v1/README.md)
removes the unnecessary forced second change.

This is a decision over fixed forecasts, not joint trajectory
generation as in [FJMP](https://arxiv.org/abs/2211.16197) or
[ScePT](https://arxiv.org/abs/2206.13387). Relative displacement
error does not validate collision safety. Intervals use 3,000
location draws after averaging seeds and calibration directions.

[Protocol](protocol.md) | [Grouped readout](readout.json)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_vehicle_angular_pair.py
```

The reader resumes per completed direction and checks the frozen
packet hash. It opens no external outcomes and trains no new model.
