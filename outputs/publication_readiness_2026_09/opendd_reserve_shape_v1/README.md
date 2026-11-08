# Fixed Reserve Shapes and the Reserve Alone

I froze the recipients and query times of the previous OpenDD
reserve policies before comparing trajectory shapes. The primary
comparison uses the 598 gain-triggered slots; the 345 easy-triggered
slots and 2,280 original edits are separate controls. At each
comparable slot, I use the same peak departure from CV: the
smallest native radius of the neural change, OLS4, and an analytic
path selected on observed prefixes. None is amplified. If one
radius is zero, every arm retains the original edit at that slot.

The analytic selector uses the same observed history to compare
OLS4, OLS3, OLS8, damped motion and bounded acceleration on
prefixes of length 4, 5 and 6. It never uses a future label
to pick a candidate. The original rows and realized counts
reproduce exactly, including the incumbent edits. Training was
not changed or rerun.

## Does the Neural Shape Help at the Same Radius?

| Frozen slot stream | Matched / all slots | Lower OLS4 vs neural, pp | Lower neural vs OLS4, pp |
| --- | ---: | ---: | ---: |
| Gain reserve | 594 / 598 | -0.084 [-0.170, -0.002] | +0.078 [-0.007, +0.168] |
| Easy reserve | 337 / 345 | +0.066 [-0.009, +0.196] | -0.068 [-0.197, +0.006] |
| Original edits | 2,215 / 2,280 | -0.440 [-0.753, -0.148] | +0.209 [-0.231, +0.560] |

The bounds are directional: swapping the policies reverses the
observed gain but still subtracts the missing-step debit. A
negative lower bound in one direction does not prove positive
advantage in the other. The gain-slot neural mean observed FDE
is 1.236 versus 1.251 for OLS4, over 573 available endpoints,
in dataset-local coordinates. The reverse conservative
comparison does not establish a reliable neural advantage.
The relative-position lower effect on the gain stream is also
unresolved (OLS4 vs neural -0.020 pp [-0.483, +0.576]).
The hindcast-selected analytic arm does not improve full-roster
lower gain over matched neural.

## What Happens If the Incumbent Edits Are Removed?

I separately replayed the gain reserve while reverting all
other rows to CV. One arm retains the original causal
eligibility trace to preserve exactly the 598 chosen rows.
A simpler arm does not use that trace: it takes the first
query with positive predicted lower gain, selects the largest
predicted gain there, and makes at most one quarter-amplitude
edit. Both use the existing head and thresholds.

| Policy | Edits | Undefined | Risk pass | Joint pass | Lower gain vs CV, pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original policy | 2,280 | 401 | 26 | 16 | -0.190 [-0.622, +0.164] |
| Original plus gain reserve | 2,878 | 288 | 77 | 56 | -0.048 [-0.427, +0.287] |
| Reserve only, frozen rows | 598 | 459 | 88 | 66 | +0.141 [+0.036, +0.239] |
| Reserve only, without shadow | 598 | 463 | 86 | 64 | +0.143 [+0.031, +0.241] |
| Reserve only, matched neural | 598 | 459 | 91 | 70 | +0.138 [+0.033, +0.233] |
| Reserve only, matched OLS4 | 598 | 459 | 74 | 57 | +0.057 [-0.114, +0.215] |

The no-shadow arm is positive at six of seven sites. Its
CV-error-mass-weighted effect is +0.051 pp, and the point and
bootstrap lower bound remain positive in each leave-one-site-out
sensitivity. These are post-hoc development checks. Average
easy net degradation is 0.35%, but it does not imply compliance:
463 views are undefined and only 86 pass original risk. Joint pass
also requires positive conservative gain against both CV and OLS4.
The 606 site-seed views repeat 202 exposed recordings. Intervals
use 3,000 site draws after averaging the three seeds.

The loss of the original edits offsets the small aggregate gain
of the reserve. That is stronger evidence for studying a sparse
intervention than for restoring the full queue. I would next test
whether reserving an independent low-amplitude support action
improves risk coverage at matched budgets, with the gain-only
one-edit policy kept as a control. I do not advance the sparse
policy to confirmation or deployment from these results.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_reserve_shape.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_reserve_only.py
.venv-pytorch/bin/python scripts/audit_m3w_opendd_reserve_shape_direction.py
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-reserve-shape
```

`fresh_run`: fixed shape and sparse-policy inference, directional
and site-sensitivity readout. `cached_verified`: existing trajectory
bank, checkpoint heads and incumbent; both causal runs reproduce
their immutable results. Thirty-four focused tests passed.
Local arm64, four threads, no loader processes, no training or
CREATE job. Original 2% rule, 8/12 annotation steps and dataset-local
coordinates. DroneCrowd confirmation was not opened; no deployment,
Stage5C or SMC.

[Shape readout](analysis.json) | [Direction and sensitivity](direction.json) |
[Sparse policy readout](../opendd_reserve_only_v1/analysis.json)
