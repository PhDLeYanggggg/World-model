# OpenDD: Where the Joint Options Disappear

The acceleration refit rarely made a different joint decision after
gating. I checked whether the policy simply missed useful choices, or
whether the admitted edit slots had left it nothing to choose.

## Exact Comparison

I reused the frozen forecasts for 202 recordings, 7 locations and
3 seeds. For each of the four release rules I fixed the exact query,
edit count and peak displacement of every slot. I left one-edit
queries at the matched additive result. At queries with two edits I
enumerated the legal assignments to distinct actors and solved the
recording-level optimization over these assignments.

Each witness must beat both CV and OLS4 on the missing-step-debited
trajectory metric, preserve the additive control's accuracy, improve
relative-position error over additive on the full grid, and pass all
three original 2% risk checks with positive reference support. The
second solve excludes all candidate combinations with the same
trajectories as the first, including equal-radius slot swaps. I replay
each witness with the original risk and trajectory metrics. These
are hindsight bounds, not causal predictions or proof of deployment
risk. There were no unresolved or timed-out recording views.

## What Changed

| Frozen release | Edits | Edited recording-seed views | Views with any two-edit query | Views with two distinct strict witnesses | Unique recordings |
|---|---:|---:|---:|---:|---:|
| Forced slots | 3,690 | 439 | 254 | 48 | 29 |
| Moment gate | 267 | 121 | 6 | 1 | 1 |
| Moment gate + support 0.25 | 222 | 101 | 6 | 1 | 1 |
| Moment gate + support 0.50 | 40 | 7 | 2 | 0 | 0 |

The 48 forced-budget views with witnesses come from three locations: rdb1 (29
seed-views), rdb5 (2), and rdb6 (17). Here is the paired breakdown on
those same 48 views, not a comparison of different subsets:

| Gate applied to the 48 forced-positive views | No edit | Only singleton queries | At least one two-edit query |
|---|---:|---:|---:|
| Moment gate | 28 | 20 | 0 |
| Moment gate + support 0.25 | 32 | 16 | 0 |
| Moment gate + support 0.50 | 48 | 0 | 0 |

The single feasible view in the first two gated arms is at rdb6 and
is not among the 48 forced-positive views. Because its budget and
reference differ, it is not a recovery of one of those opportunities.
The learned policies had zero realized qualified positive pair wins
in the [previous replay](../opendd_supported_acceleration_v1/README.md).

## Consequence For the Method

On this development cohort the main loss happens at **admission**:
the gate suppresses every previously feasible two-edit opportunity
before pair ranking. Refitting the pair ranker alone cannot improve
these views at their admitted slots. The next intervention should
score entire two-edit releases, including the chance of both
accuracy and reference support, before committing to single edits.
Compare against additive at the same released slots and assess the
release mechanism separately at matched intervention counts. Do not
use these hindsight views to tune the risk limit.

Paper-ready summary: *At the same intervention budget, a joint
decoder can only help when the release retains alternative
assignments. Under forced slots 29 OpenDD development recordings
admitted multiple verified improvements; after evidence-gated
release this fell to one or zero, even with hindsight.*

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/audit_m3w_opendd_distinct_supported_slates.py
```

Uses the location-excluded acceleration [parent run](../opendd_supported_acceleration_v1/README.md).
The arm64 Mac run took 25.21 seconds and 811 MB maximum RSS, no
CREATE job or new training. Each completed fold has a hash-bound
checkpoint and phase/PID heartbeat. Recomputed with a new Python
bytecode-cache directory and no cache writes: all 21 fold records
matched byte-for-byte. Seventeen focused tests passed.
[Registration](registration.json) and [summary](summary.json).
No independent confirmation outcomes were opened. Stage5C and SMC
remain off.
