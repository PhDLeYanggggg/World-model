# AV2: Improving the Initial Reference by Steering Two Fixed Edits

The [direction and timing comparison](../av2_direction_menu_v1/README.md)
found a matched alternative for 62 of 63 eligible references. I
now try to improve the reference itself, without changing its two
recipients or spending more displacement.

## Design

Each edited actor gets the same small network. It encodes eight
observed positions, CV, the original CV-relative offset, and OLS4
in a history-aligned frame. Three predicted angle knots are
interpolated across the twelve forecast steps and bounded to
45 degrees either side. Rotating each offset preserves its norm
at every step; unedited actors remain at CV. Both heads start at
zero steering and see identical minibatches and initial weights.

The ADE head minimizes normalized known-position error. The guarded
head adds the negative portions of the two conservative gains and
positive excesses of the three original risk terms, with weights
1 and 2 respectively. Missing-step disagreement is debited in both
motion comparisons. Unchanged actors still contribute to the OLS4
reference and signed easy reference mass. The soft training loss
is not a qualification test; the exact original scorer is used
after prediction.

For each held city and seed, I fit only on the four donor cities.
Donor incumbents come from the existing models that excluded the
donor's own city, the outer calibration city and the held city.
Neither calibration nor held labels fit the steering heads. The
18 paired fits (36 heads) use 300 shared updates per fold. Held
paths and checkpoints are hashed before scoring. This test measures
forced standalone qualification, not a calibrated release policy.

## Results

| Reference | Both motion floors and risk | Fully labeled passes / 465 | Risk pass / 1,020 | Known ADE / available FDE |
|---|---:|---:|---:|---:|
| Original independent | 63 | 43 | 109 | 0.2480 / 0.5518 |
| ADE steering | 62 | 43 | 112 | 0.2486 / 0.5533 |
| Guarded steering | 62 | 43 | 110 | 0.2505 / 0.5574 |

Guarded steering rescues two views and loses three. Its direct
conservative gain relative to the original is **-1.128 pp
[-1.616, -0.676]**; ADE steering is **-0.374 pp
[-0.592, -0.152]**. The worst recording/scenario-view p95 ADE
increases from 5.9291 to 6.2249 and 6.3362 native units. The guarded
head also worsens the pair error relative to the original. The 1,020
views are 340 development scenarios repeated across three seeds.
Intervals average seeds within scenarios and resample six cities
3,000 times. Neither training objective improves the reference.

### Exact relaxed direction ceiling

At a fixed CV-relative radius, the point on the circle closest to a
known future position lies toward that position. On an unobserved
step, aiming toward OLS4 minimizes the OLS4 disagreement debit and
cannot change the CV debit. For a bounded angular arc, clamp the
angle to the closest point in that arc. Each choice simultaneously
improves or preserves the standalone utility and risk terms; the
CV-defined easy set and edited support do not change. This gives an
exact ceiling for the relaxed problem with independent angles per
step, not a feasible motion or learned policy.

| Fixed recipients and per-step radii | Standalone ceiling | Fully labeled ceiling / 465 |
|---|---:|---:|
| Free per-step directions within ±45 degrees | 85 | 55 |
| Free per-step directions without angle limit | 176 | 116 |

The learned heads leave some bounded-direction headroom, but even
the unrestricted oracle cannot qualify most views. Only **382/1,020**
views have edited easy reference support at the fixed recipients;
steering cannot create it. The next comparison should quantify and
test recipient reassignment at the same two counts and exact radial
profiles, rather than widen the steering model or change the 2% rules.

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_steered_reference.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_steered_reference.py score
.venv-pytorch/bin/python scripts/run_m3w_av2_steered_reference.py replay
.venv-pytorch/bin/python scripts/audit_m3w_av2_steered_reference_ceiling.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py \
    tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

Native arm64, four Torch threads and zero workers. Training and freeze
took 18.59 s at 0.58 GB maximum RSS; scoring 4.85 s; ceiling 4.10 s.
Atomic checkpoints every 50 updates store optimizer and sampler state.
Replay uses scalar receipts. The focused suite passed 80 tests.
No new CREATE job, reserved outcomes, deployment, manuscript, Stage5C
or SMC change. Forecasts use native units and the same 8/12 step grid.

[Registration](registration.json) · [Frozen fits](freeze.json) ·
[Results](summary.json) · [Direction ceiling](direction_ceiling.json)
