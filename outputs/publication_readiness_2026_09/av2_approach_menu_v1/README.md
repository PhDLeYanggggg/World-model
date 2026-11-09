# Argoverse 2: Acceleration Adds Useful Choices, but Selection Still Falls Short

The first Argoverse 2 run left only five fully labelled two-edit scenes
with a feasible joint improvement, even with hindsight. I kept the
same scenes, reference forecasts, two edit slots and displacement
limits, and changed the candidate paths rather than the selector.

## Candidate Test

I added two own-history paths: a decaying acceleration estimated from
the last four past increments and a gradual blend toward the longer
past velocity. A separate two-path arm adds longitudinal braking and
lateral avoidance based on the nearest predicted approach under the
last-step velocities. All paths use only the same eight observed
positions. A path is eligible for an edit slot only if its offset can
be scaled down to the slot's original peak. The original OLS4 and
tapered-turn paths remain available in every arm.

| Candidate menu | Mean legal assignments | Pair-only wins | Pair and accuracy wins | Wins with all three risk checks | Unique scenes with strict wins |
|---|---:|---:|---:|---:|---:|
| Original | 50.2 | 73/81 | 42/81 | 15/81 | 5/27 |
| Add own-history paths | 215.0 | 78/81 | 60/81 | 38/81 | 13/27 |
| Add nearest-approach paths | 53.8 | 73/81 | 42/81 | 15/81 | 5/27 |
| Add both | 221.7 | 78/81 | 60/81 | 38/81 | 13/27 |

The original counts replayed exactly. A strict hindsight win improves
pair-relative error, beats CV and OLS4, is no worse than additive ADE,
and passes the unchanged 2% checks. Eight new scenes qualified across
four cities. The paired, equally city-weighted increase is **27.17 pp**
(95% city bootstrap interval **11.43 to 42.17 pp**). Decaying acceleration
accounts for 59 of the 76 selected slots in the own-history witnesses;
none uses the smoothed velocity. Only three of 81 seed-views had a
detected nearest-approach neighbor. That arm did not help here; this
sample cannot test it well in dense approaching interactions.

## Frozen Policy Replay

The own-history menu was chosen for this follow-up after the hindsight
readout. I replayed all 128 development scenes, not only the complete
ones, with the three existing OpenDD donor ensembles. The original
predictions reproduced exactly. I did not refit scores or change
thresholds. Every policy uses the same 459 edits across the same 285
edited seed-views.

| Policy | Complete ADE | Available FDE | Full-grid gain vs CV (pp) | Risk passes / edited views | Easy net degradation over 2% |
|---|---:|---:|---:|---:|---:|
| CV | 0.2627 | 0.5773 | 0.00 | No edits | 0 |
| Original additive | 0.2712 | 0.5880 | −3.47 | 2/285 | 28 |
| Original relative joint | 0.2714 | 0.5886 | −3.57 | 3/285 | 33 |
| Own-history additive | 0.2673 | 0.5832 | −1.88 | 5/285 | 22 |
| Own-history relative joint | 0.2669 | 0.5832 | −1.80 | 10/285 | 27 |

The direct paired comparison of the new joint policy against the old
one is **+1.70 pp** on known steps [1.22, 2.18] and **+0.57 pp** after
debiting every missing step [−0.47, 1.38]. At the same menu and budget,
joint over additive has a full-grid pair gain of +0.000821
[−0.001000, +0.004013] in native coordinates. On 59 fully labelled
scenes that contrast is +0.000579 [+0.000017, +0.001527].
These intervals use 3,000 paired draws over six cities, after averaging
seeds within each scenario. The complete-only signal is a development
result, not proof of a reliable all-scene gain.

There are **zero** realized pair wins that also pass the accuracy and
risk requirements. In 242 of 285 edited views the original edited-easy
denominator is still missing; this is not counted as a pass. The best
motion reference remains CV. The main gain here is an expanded set of
useful paths, not a new deployment policy or an independently confirmed
joint improvement.

## What To Keep

The acceleration path is worth testing in a location-excluded donor
refit. The decisive comparison should keep the same expanded menu for
additive and joint selection, and test support-aware release before
spending more training on the pair ranker. A useful next result must
improve more than hindsight capacity or the complete-only slice.
The strong CV reference is consistent with the caution in
[Schöller et al.](https://arxiv.org/abs/1903.07933); projected closest
approach is related to anticipatory interaction models such as
[Karamouzas et al.](https://arxiv.org/abs/1412.1082). Neither elementary
motion formula is a novelty claim.

## Reproduce

Requires the [parent AV2 freeze](../av2_native_trajectory_v1/README.md).
Run locally in the native arm64 environment:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_approach_menu.py freeze
.venv-pytorch/bin/python scripts/run_m3w_av2_approach_menu.py score
.venv-pytorch/bin/python scripts/run_m3w_av2_approach_replay.py freeze
.venv-pytorch/bin/python scripts/run_m3w_av2_approach_replay.py score
.venv-pytorch/bin/python scripts/report_m3w_av2_approach_menu.py
```

Frozen replay took 9.26 seconds and scoring 2.40 seconds on the Mac,
under 460 MB maximum RSS. There are atomic per-scenario forecasts and
a phase/PID heartbeat. No CREATE job, new checkpoint training, AV2
validation/test access, reserved confirmation access, Stage5C or SMC.
The focused test suite passed 27 checks. See the
[capacity registration](registration.json), [replay registration](replay_registration.json)
and [lightweight summary](summary.json).
