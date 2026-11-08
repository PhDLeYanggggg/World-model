# Ranking First-Offer Actions at Identical Radius

The [timing experiment](../radius_opening_v1/README.md) found no new
joint witness by waiting for a second offer. I hold the first offer
and its radius fixed, then separate shortlist recall from the actual
one-action decision. All four opening arms make the same 29 edits
at the same per-recording radii; the incumbent without an opening
is reported separately.

Both learned heads see the same causal descriptors, peer geometry,
forecast direction and frozen cost scores. Each fits the parent
study's marginal recording quality. The listwise variant adds a
within-query softmax loss to the same pointwise regression. The
models use identical initialization and draws (8 offers of 64
sampled actions per update, 300 updates). This is an application of
[listwise ranking](https://doi.org/10.1145/1273496.1273513) to a
fixed trajectory choice, not a new generic ranking method.

| One-action policy | Openings | Risk passes / 102 | Undefined | Joint |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 0 | 2 | 29 | 0 |
| Frozen rank | 29 | 5 | 25 | 0 |
| Ridge rank | 29 | 3 | 26 | 0 |
| Pointwise Torch | 29 | 2 | 28 | 0 |
| Listwise Torch | 29 | 3 | 24 | 0 |

The listwise-minus-pointwise contrast is **-0.00361 pp
[-0.00644, -0.00112]** in paired conservative utility. The
contrast against frozen rank is -0.00289 pp [-0.00762, +0.00035].
Pointwise gains +0.00327 pp [+0.00039, +0.00682] over no opening,
but does not improve the full risk count or create a joint witness.
Intervals use 3,000 paired resamples of 14 sources, averaging
three seeds first. The 102 views are from 34 recordings.

## Shortlist Diagnostic

The outcome-aware full first-offer menu covers six joint views.
At fixed shortlist size, reorder by taking one candidate per
recipient or one per displacement sector before filling the
remaining slots from the original rank. Counts are views with a
joint witness somewhere in the slate, not policy successes.

| Rank | Flat top 1 | Flat top 8 | Recipient top 8 | Direction top 8 | Flat top 16 | Recipient top 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen | 0 | 1 | 0 | 0 | 1 | 0 |
| Ridge | 0 | 1 | 2 | 1 | 2 | 3 |
| Pointwise | 0 | 0 | 0 | 1 | 0 | 1 |
| Listwise | 0 | 1 | 2 | 2 | 2 | 2 |

The v1 run stopped on a fold with no fitting offer after 10,800
updates. Registered v2 preserves that failed run; it skips model
fitting on six empty donor cells and uses the frozen rank for both
arms. Three other cells have offers from only one donor; four have
offers from all seven. No held or reserved source supplies training
examples. With few admissible opportunities and only six joint
witnesses, this is not a useful target for scaling another ranker.
Next prioritize a higher-support development setting such as
[the matched TUMDOT transport menu](../../../tumdot_matched_finite_transfer_v1/README.md):
there were feasible positive/risk witnesses across all 27 repeated
views, while fixed causal forecasters were still negative.
Any TUMDOT tuning on those views stays development, not confirmation.

Original 2% rule and undefined handling stay unchanged. No deployment,
independent confirmation, Stage5C, SMC, or new metric/seconds claim.
DroneCrowd stays at 15 calibration and 30 sealed confirmation groups.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opening_slate.py
.venv-pytorch/bin/python scripts/run_m3w_opening_slate.py --replay
.venv-pytorch/bin/python scripts/build_m3w_opening_slate_replay.py
```

`fresh_run`: 72 fits, 21,600 updates, 44.557 seconds, PID 89522.
Four native arm64 Torch threads, zero loader workers, checkpoint and
heartbeat every 100 updates. Rerun the first command to resume.
`cached_verified`: frozen parent predictions and exact zero-update
replay. `not_run`: CREATE or independent confirmation.
