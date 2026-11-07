# A Learned Motion Gate Does Not Improve The Matched Policy

Six native arm64 CPU PyTorch fits use 69,746 windows from nine
nonempty fitting source groups, excluding all 14 development
sources. Each head has 3,560 parameters and trains for 900 updates
at batch size 128, using identical initial weights and draw
sequences within each seed. The ungated arm does not use the two
modulation outputs. The heads predict bounded longitudinal
and lateral velocity adjustments around OLS4; only the gated
head modulates the adjustment with learned per-axis factors.
Training took 11.1 seconds including fitting-data preparation,
with two Torch threads and no loader workers. Both final weights
and all six past-only forecast arrays were frozen before scoring.

| Development contrast | Available labels (pp) | Conservative lower change (pp), 95% CI |
| --- | ---: | ---: |
| All-row gated vs. OLS4 | +15.244 | +2.018 [-12.256, +15.573] |
| All-row gated vs. CV | +11.286 | -3.311 [-21.534, +11.702] |
| All-row gated vs. ungated | +0.736 | -2.535 [-6.039, +0.298] |
| Fixed recipients, ungated vs. original | -0.239 | -0.797 [-1.777, -0.060] |
| Fixed recipients, gated vs. original | -0.291 | -0.905 [-2.020, -0.049] |
| Fixed recipients, gated vs. ungated | -0.052 | -0.243 [-0.590, +0.010] |

The denominator is observed CV error mass, not the comparator's
own error. The intervals use 3,000 paired source resamples,
averaging the three seeds first. Both heads replace the same
9,675 already switched paths. The original, matched ungated
and matched gated policies pass 4/5/4 of 102 repeated views.
Their worst signed easy degradation is 98.70/24.48/25.78 percent,
still far above 2 percent. Worst recording p95 available-label
ADE is 288.10/289.04/289.04 image pixels. Both all-row forecasts
pass zero recording checks.

The separate [observed slice analysis](matched_slice_analysis.json)
uses only the originally switched subset and normalizes by the
observed CV mass of each slice. On 2,827 easy switched rows,
ungated and gated have positive point estimates versus original
(+28.87 and +21.73 pp) but intervals span zero. The gated arm
does not beat the ungated arm on easy rows (-7.14 pp, 95% CI
[-14.07, -0.26]). Eight required hard source/seed cells lack
positive reference mass; the full source-balanced hard contrast
is not estimable. These are repeated forecast instances, not
independent people or scenes.

This is not a reason to promote either learned head or the
gate. The all-row observed benefit does not carry to the actual
fixed-recipient intervention. The separate
[optional-menu audit](../native_gated_motion_menu_v1/README.md)
tests whether the trained paths are nonetheless useful as
additional candidates in hindsight.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_gated_motion_residual.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_gated_motion_residual.py infer
.venv-pytorch/bin/python scripts/run_m3w_gated_motion_residual.py score
.venv-pytorch/bin/python scripts/run_m3w_gated_motion_residual.py replay
.venv-pytorch/bin/python scripts/verify_m3w_gated_motion_resume.py
.venv-pytorch/bin/python scripts/report_m3w_gated_motion_slices.py
```

[Protocol](protocol.md) | [Registration](registration.json) |
[Training receipt](train_execution.json) | [Forecast freeze](infer_execution.json) |
[Analysis](analysis.json) | [Score receipt](score_execution.json) |
[Replay receipt](replay_execution.json) | [Resume proof](resume_verification.json).
Ten focused tests passed. A separate real-data seed-17 gated run
resumed from update 150 and exactly reproduced the final weights,
optimizer, random states, draws and training trace at update 900.
Checkpoints and forecast arrays remain local. Native eight/twelve
annotation steps and image pixels are not metric or seconds.
There are no independent outcomes, deployment, Stage5C or SMC.
