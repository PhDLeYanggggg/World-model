# AV2: A Matched EqMotion Control and Three Edit Policies

I trained a public forecaster on the expanded AV2 development
roster to test whether the bottleneck was the trajectory bank
or the way the policy selects and combines its forecasts.

## Public Forecast Control

I used the verified author core of [EqMotion, Xu et al., CVPR 2023](https://arxiv.org/abs/2303.10876)
at [commit 5aec2e0](https://github.com/MediaBrain-SJTU/EqMotion/tree/5aec2e0b61c511fa93a24138dd90da59a089084b).
The compact centered-peer configuration uses eight observations,
twelve requested positions, up to three observed neighbors and
one output head fixed before training. It is a single-forecast
adaptation, not a reproduction of the paper's best-of-20 results
or a leaderboard comparison. No pretrained weights were used.

The six AV2 city folds and 896 added evaluation scenarios are
identical to the [data-scale comparison](../av2_data_scale_v1/README.md).
Only the original four donor cities fit each fold. Both the
calibration city and the held city are excluded. Three seeds
each make 800 optimizer updates; trajectories are masked for
the loss only. The original donor-derived easy and hard cuts
are retained for scoring. Predictions are frozen before scoring.

The raw forecaster predicts for all agents. For the forced
intervention comparison, it contributes just one extra path
family to the existing menu. A path can fill a slot only if
it can be scaled down to its original peak offset. The original
options and recipient roster stay in the menu. The old scores
and joint decoder are unchanged, and the old forecasts replay
exactly. The three arms on this expanded menu are:

- **Scene uniform:** both slots must use the same family; use
  the same additive cost forecasts to choose the assignment.
- **Independent additive:** allow different families and sum
  the same per-action scores without a pair reward.
- **Joint:** use the same candidates and budget, with the
  frozen scene-level risk and relative-gain re-ranking.

## Accuracy and Risk

| Forecast | Known ADE | Available FDE | Conservative gain vs CV (pp) | Hard gain vs CV (%) | Risk passes / edited views |
|---|---:|---:|---:|---:|---:|
| CV, no edits | 0.24668 | 0.57458 | 0 | 0 | No edits |
| OLS4 | 0.27864 | 0.61665 | -22.03 | -10.91 | 9/2,688, unmatched |
| Raw EqMotion | 0.34732 | 0.66486 | -87.46 | -10.42 | 1/2,688, unmatched |
| Original additive | 0.24510 | 0.56835 | -0.81 | +0.71 | 115/1,932 |
| Original joint | 0.24487 | 0.56783 | -0.76 | +0.71 | 117/1,932 |
| EqMotion menu, scene uniform | 0.24573 | 0.56897 | -1.01 | +0.59 | 100/1,932 |
| EqMotion menu, independent additive | 0.24582 | 0.56924 | -1.06 | +0.59 | 110/1,932 |
| EqMotion menu, joint | 0.24578 | 0.56899 | -1.04 | +0.57 | 105/1,932 |

The five forced policies have 2,952 edits at identical query
peaks. Distances use AV2 native coordinates. For the three
expanded-menu arms, undefined edited-easy ratios are 1,659,
1,666 and 1,655; signed easy degradation exceeds 2% in 139,
118 and 132 views respectively. Mean scenario p95 ADE is
0.68656, 0.68688 and 0.68598; for raw EqMotion it is 0.82813
versus CV's 0.69153. The worst observed scenario-seed p95 is
9.396 for raw EqMotion, 7.432 for CV and 7.295 for expanded
joint. No arm beats CV on conservative trajectory
gain; raw EqMotion's interval is [-147.87, -47.38] pp. Known
ADE/FDE and conservative gain use different treatment of
missing steps and different aggregation.

| Direct full-grid pair contrast | Gain in native units [95% interval] |
|---|---:|
| Independent additive vs scene uniform, same EqMotion menu | -0.002601 [-0.003852, -0.001463] |
| Joint vs scene uniform, same EqMotion menu | -0.003400 [-0.004854, -0.001943] |
| Joint vs independent additive, same EqMotion menu | -0.001599 [-0.002291, -0.000989] |
| New joint vs original joint, same peaks | -0.002900 [-0.004062, -0.001684] |

The intervals use 3,000 paired city bootstrap draws after
averaging seeds within scenario. On known steps only, joint
vs independent is +0.000038 [-0.000526, +0.000628]. On 422
fully observed scenarios it is -0.000055 [-0.001241, +0.001336].
The full-grid deterioration is not evidence of a reliable
known-step benefit.

## Capacity and Actual Selection

| Two-slot outcome | Original menu | Add EqMotion |
|---|---:|---:|
| Legal assignments across 1,020 held seed views | 230,907 | 411,251 |
| Any strict gain versus the fixed original additive path | 252 | 271 |
| Any two legal paths can both qualify with pair lift | 300 | 334 |
| Frozen joint decoder chooses a strict gain versus its additive path | 9 | 9 |
| Both selected arms qualify | 2 | 2 |

The first three rows describe hindsight capacity, not a policy.
Even among the 465 fully labelled two-slot views the two-arm
ceiling rises from 183 to 204, but both selected arms qualify
only twice. The new family is used, not merely present: 466
additive and 569 joint edits use a path that the original menu
could not generate. The joint usages span 339 unique scenarios.
The comparison separates a real increase in candidate capacity
from a failure to recover it with the old decoder.

I would keep EqMotion as a matched public control, not as a
replacement for CV. The next discriminating test is to train
selection on out-of-fold neural candidates with dual accuracy
and easy support labels. The same split, uniform control and
exact-slot budget should carry over. A positive result still
needs nonempty calibration and untouched external confirmation.

## Reproduce

Use the [AV2 source and freeze](../av2_data_scale_v1/README.md)
and the pinned author source, fetched with
`scripts/fetch_m3w_eqmotion_source.py` if absent.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_eqmotion.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_eqmotion.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_eqmotion_choices.py
.venv-pytorch/bin/python scripts/run_m3w_av2_eqmotion_uniform.py freeze
.venv-pytorch/bin/python scripts/run_m3w_av2_eqmotion_uniform.py score
```

Native arm64, four Torch threads, zero loader workers. The 18
fits and forecast freeze took 308.89 s; scoring took 25.02 s,
below 932 MB RSS. The same fit command resumes from an atomic
checkpoint every 50 steps. The heartbeat records the fold,
step and process ID. Thirty-one focused tests passed.
No CREATE job, new risk calibration, reserved confirmation
access, Stage5C, SMC, deployment or manuscript change.
[Protocol](registration.json), [metrics](summary.json),
[scene-uniform contrast](uniform_summary.json),
[shape usage](usage.json).
