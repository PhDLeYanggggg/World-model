# Does a Better Turn Forecast Help Joint Intervention?

I added a tapered turn to the OpenDD forecast menu and retrained the
gain and harm heads. The turn is extrapolated from the eight observed
positions, using the [existing causal turn model](../radius_matched_turn_menu_v1/README.md).
It adds a path, not an extra edit. I held the 12 forecast steps, candidate
recipients, per-query edit counts and peak displacements, and original
recording-level 2% risk rule fixed. All four fitted policies use the same
original additive output on singleton queries, so only the two-edit
choices can explain a difference.

## Capacity and Realized Gain

The development set has 202 recordings at seven locations, with three
seed views per recording. Of 606 views, 439 have edits, for a total of
3,690 fixed slots in each arm. Only 1,067 queries have two edits. The new
path is selected uniquely in 760 additive and 781 joint slots in those
queries, so the learned result is not an unused-candidate test.

| Forecast menu | Co-observed strict positive views | Full-grid strict positive views | Distinct full-grid positive recordings |
|---|---:|---:|---:|
| OLS4, median slope and held-source neural path | 25 | 21 | 13 |
| Same menu plus tapered turn | 55 | 50 | 32 |

The table is **hindsight**, not model performance. The exact solver uses
future labels to choose paths, while keeping singleton decisions and the
original additive reference fixed. A witness must pass the original
risks, beat both CV and OLS4 conservatively, and preserve conservative
individual accuracy versus that additive reference. Missing pair labels
debit the reverse-triangle bound on the full grid. The accuracy
prescreen was recomputed on the larger menu, not inherited from the
narrower one. All 162 interaction solves returned optimal, independently
rescored witnesses. No old verified optimum was lost. The 50 positive
views are concentrated at three sites: 28 at rdb1, four at rdb5 and 18
at rdb6.

| Fitted policy (fixed singleton choices) | Risk passes | Risk pass and beat CV/OLS4 | Observed ADE | Available-endpoint FDE |
|---|---:|---:|---:|---:|
| Previous menu, additive | 39/439 | 34/439 | 0.54861 | 1.25273 |
| Previous menu, joint | 39/439 | 34/439 | 0.54861 | 1.25274 |
| Turn menu, additive | 42/439 | 38/439 | 0.54830 | 1.25312 |
| Turn menu, joint | 46/439 | 40/439 | 0.54840 | 1.25340 |

The ADE and FDE columns are recording-seed means in native units, not
metres. Only **3/439** views have positive conservative pair gain, pass
all risks and both individual baselines, and do not lose individual
accuracy versus the new policy's own additive control. They come from
three recordings at rdb1. In this fitted comparison the full-grid
lower gain is **-0.195 [-0.416, -0.009]** in 10^-3 native units;
the co-observed interval spans zero (**-0.091 [-0.238, +0.024]**).
Intervals use 3,000 paired location draws from the six estimable
locations, after averaging seeds within recordings. A negative lower
bound does not establish that unobserved pair errors increased. The
hindsight and fitted wins have different additive references; one is
not a recall denominator for the other. No-edits views are not passes.

## What This Changes

The larger menu gives the method something better to choose, but the
current joint rule still aggregates per-action gain and harm. Its loss
does not supervise the relative-displacement effect of the entire slate.
The next development comparison should hold the larger menu and all
slots fixed and train on the conservative pair effect of matched
alternative slates, with additive and safety-only joint controls. A new
test source must have independent provenance and enough co-observed pair
and easy-reference support; an exposed development recording cannot
confirm that result.

[M2I](https://arxiv.org/abs/2202.11884) and
[FJMP](https://arxiv.org/abs/2211.16197) model conditional joint futures.
This experiment instead asks when a fixed forecast replacement earns
its edit under matched costs. That is a different comparison, not proof
of superiority over either method. Nor does an empirical recording
check establish the expected-loss guarantee of
[conformal risk control](https://openreview.net/pdf/482dcb57a6da0d60fce0a3b533b80bcbe99b03f6.pdf).
I would use this result in a limitations or mechanism analysis, not as
the paper's main effectiveness claim.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_turn_intervention.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_turn_intervention.py score
```

The 21 fresh CPU fits used native arm64, four compute threads, zero
workers and 400 optimizer steps per fit. Each fit excludes its scored
location; the donor forecasts additionally exclude the donor location
from their own fit. The first command saves weights, optimizer and
sampler state every 50 steps and freezes forecasts before scoring. The
second checkpoints each fold and verifies parent, forecast and row
bindings. Rerun the interrupted command to resume. PID and progress
are in the local experiment directory's `heartbeat.json`.
Checkpoints and trajectories stay local. No CREATE submission, reserved
outcome access, threshold change, Stage5C, SMC or deployment change.

[Registration](registration.json) | [Freeze receipt](freeze.json) |
[Readout](readout.json)
