# First-Edit Timing at Identical Displacement

The [exchange study](../small_support_exchange_v1/README.md) could
not repair views in which the incumbent never acted. This test
compares the timing of one additional edit, with the displacement
radius held fixed and the original prediction stream retained.
Unlike the [prior stop/wait study](../opening_stop_wait/readout.md),
the primary comparison matches actual edit count and radius within
recording, and the candidate labels refer to the full recording's
marginal utility and risk rather than to standalone edit success.

## Policy and Controls

Before the first incumbent action, a query can offer a forecast edit.
Set the recording's radius to one quarter of the median nonzero
candidate radius at its first available query; scale candidates down
to that radius and exclude candidates too small to reach it.
Consider only the first two eligible offers. Both timing arms share
the same fitted action ranker. A separate ridge head predicts whether
the second offer is worth waiting for from first-offer features only.
If the incumbent acts first or no second offer arrives, the wait
expires. Subsequent incumbent actions are never moved or replaced.

Fit on seven sources per cell. To label timing, pick the action with
a ranker trained on the other six fitting sources and measure each
action's marginal quality, with zero change if waiting expires.
The quality combines joint success, change in risk support and
radius-normalized conservative gain; future labels enter only
supervision or scoring. Six reserved sources are not used to
fit or choose thresholds; one held source is scored per cell.
The original frozen incumbent is held-source excluded, but its
decisions on donor records are not cross-fitted.

The matched first-offer arm opens only where the learned timing arm
eventually opens. It is an offline utility control, not an online
rule. All full policy scores retain every recording and every
expired wait. A joint witness requires the original risk checks
and positive paired conservative utility against the incumbent,
CV and OLS4. Joint counts for the unchanged incumbent are zero
by construction because it cannot strictly improve against itself.

| Rule | Opened / late | Risk passes / 102 | Undefined | Joint |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 0 / 0 | 2 | 29 | 0 |
| First, frozen rank | 29 / 0 | 5 | 25 | 0 |
| First, learned rank | 29 / 0 | 3 | 26 | 0 |
| Always wait | 23 / 23 | 4 | 25 | 1 |
| Learned stop | 25 / 11 | 3 | 26 | 1 |
| First, matched to learned stop | 25 / 0 | 3 | 26 | 0 |

Learned timing versus the matched earliest control is **-0.00006
pp** of conservative utility, 95% CI **[-0.00286, +0.00285]**.
Without matching actual opens, the contrast is -0.00087 pp
[-0.00384, +0.00211]. The learned arm's +0.00277 pp
[+0.00005, +0.00596] versus no additional opening is a small
development signal, not an advantage over the count-matched
control or a safety pass. It opens all 12 zero-incumbent views but
still leaves 26 views undefined across the full roster.
Four waits expire. No policy was promoted.

## What Failed

Of 102 views, 29 have a first offer and 23 a second. Across all
candidate actions at the registered radius, six views have a joint
witness at the first offer and still six have one over both offers.
The learned top-one action on either offer recovers only one.
Increasing the timing model is not the priority for this menu.
The direct next comparison is candidate/action quality at a fixed
early query and displacement, with a simple diverse menu as a
control, or a better-supported development domain; waiting alone
does not add outcome-aware capacity here.

Intervals resample 14 source groups 3,000 times after averaging
seeds. The 102 views are repeated from 34 recordings. The original
2% rule and undefined handling remain. DroneCrowd still has 15
calibration groups and 30 sealed confirmation groups. No Stage5C,
SMC, deployment, or new metric/seconds claim.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_radius_opening.py
.venv-pytorch/bin/python scripts/run_m3w_radius_opening.py --replay
.venv-pytorch/bin/python scripts/build_m3w_radius_opening_replay.py
```

`fresh_run`: 42 source-held coefficient fits on native arm64,
41.305 s, PID 88574; no new forecaster training. Per-cell coefficient
and scalar checkpoints plus heartbeat; repeat the first command to
resume. `cached_verified`: frozen prediction lineage and exact
coefficient/score replay. `not_run`: CREATE or independent confirmation.
