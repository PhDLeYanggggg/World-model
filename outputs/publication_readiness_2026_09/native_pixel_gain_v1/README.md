# Past video motion did not improve forecast replacement

I tested whether the observed video adds useful evidence for replacing
a forecast. This differs from earlier camera-frame experiments: the
native forecast is never remapped. All arms use the same OLS4 floor,
quarter-amplitude candidate, frozen support probabilities and exact
counts in predicted-support quartiles. Only the gain head's inputs
change: no image, background motion, or background plus local pixels.

## What was measured

The exact fitting roster needs 1,344 observed frames in 24 recordings.
996 rasters are verified reuse; 348 are newly acquired from the pinned
publisher archive. The new range transfer is 673,282,302 bytes in 162
requests, taking 109.3 seconds. Independent local verification checks
all 1,344 frame receipts with zero new network reads. No raw JPEG is
retained and no future image or outcome enters feature extraction.
[Acquisition scope](../native_fit_past_pixels_v1/protocol.md)

The background estimator supports 1,078 of 1,320 transitions.
Local optical flow uses five samples seeded inside each previous
head box and a forward/backward consistency check, not a fit to the
next box. Across 69,746 windows, 99.32% have at least one accepted
local transition and 87.51% have the last transition accepted.
Acceptance is algorithmic coverage, not verified head identity:
the small patch can include background or another agent.
All rows, including unsupported measurements, remain in the
experiment. The original trajectory geometry hash and both image
feature arrays reproduce exactly in a separate process.

## Fixed-support comparisons

81 fresh Torch fits, three seeds, nine held fitting sources and 500
steps per fit. Every active arm makes 2,352 changes per seed across
1,176 queries, with the same predicted-support quartile counts.
The 3,000 paired resamples average seeds within source first.
Gains are points on the observed CV error denominator.

| Direct contrast | Observed gain | Full-grid lower gain [95% CI] |
| --- | ---: | ---: |
| Background vs no image | -0.0064 | -0.0408 [-0.0621, -0.0208] |
| Foreground vs background | -0.0029 | -0.0181 [-0.0278, -0.0092] |
| Foreground vs no image | -0.0093 | -0.0477 [-0.0746, -0.0234] |
| Foreground vs OLS4 | +0.1345 | +0.0402 [-0.0649, +0.1320] |
| Foreground vs minimum change | +0.1175 | +0.0208 [-0.0683, +0.1090] |

Conditional error is 0.6453 without images, 0.6790 with background
and 0.6767 with background plus local pixels. The foreground arm
changes 467 to 507 queries per seed relative to no image. It is
not an identity control, but the new choices show no benefit.
Foreground ADE/FDE on available labels are 91.473/234.250 pixels
versus 91.597/234.452 for OLS4. Its hard contribution is -0.0183
points. Worst supported known-easy positive harm against CV is
261.01%-261.39%; one source has undefined easy support. This does
not pass the original recording rule. The conservative lower
contrasts do not identify actual error on missing labels.

This rules out a useful gain from this fixed low-resolution
background/local-flow representation on the tested roster, not
from all past-video cues. The evidence does not justify widening
this selector again on the same sources. Priority is a better
supported independent source and a stronger forecasting comparison
before opening confirmation. The improvement in label-support
prediction remains separate from the failed decision comparisons.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_fit_past_pixels.py prepare
.venv-pytorch/bin/python scripts/run_m3w_fit_past_pixels.py verify
for phase in features feature_replay train score replay; do
  .venv-pytorch/bin/python scripts/run_m3w_pixel_gain.py "$phase"
done
.venv-pytorch/bin/python scripts/render_m3w_support_utility.py
```

Use `prepare --resume` or `train --resume` after checking the
original process has ended. Features and feature replay each
take about 23 s; training takes 33.7 s. Native arm64 CPU, four
Torch compute threads, no loader workers, checkpoints every
100 updates. Twelve targeted tests pass. Feature and score
replays match exactly. The aggregate plot is rebuilt identically
from public JSON; local figure files, images and weights are not
in Git.

[Protocol](protocol.md) | [Features](feature_report.json) |
[Results](analysis.json) | [Replay](replay_execution.json).
Fitting-source evidence only. Original 2% rule unchanged, 30
DroneCrowd confirmation groups sealed, no deployment,
Stage5C, SMC, metric or seconds claim.
