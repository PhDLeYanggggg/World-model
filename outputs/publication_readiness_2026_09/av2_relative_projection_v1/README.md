# AV2: projected position reference transfers across development cities; relative refinement does not

The [calibration comparison](../av2_projected_reference_v1/README.md)
found that learning direction at a fixed radius improved standalone
qualification. I froze that position model and tested whether supervising
relative trajectories produces an additional gain at the same budget.

## Matched experiment

The original soft reference determines exactly two recipients and each
offset's magnitude at every step. The frozen position model changes only
directions. Two new heads start from that identical projected forecast and
rotate the offsets by at most 45 degrees using three angle knots. Both see
the two agents' observed histories, relative geometry and the frozen
projection. Their architecture, initial weights and sampled donor options
are identical. The difference is supervision:

- **Individual refinement:** known trajectory error and missing-step
  disagreement versus the frozen position projection.
- **Relative refinement:** the same individual loss plus 0.75 times the
  pair loss. This includes every edge touched by an edit, including edges
  to unedited actors. A missing edge-step is charged the change from the
  frozen reference. The loss difference replays the original evaluation
  arithmetic, with a common full-grid denominator.

There are four donor cities, a separate calibration city, and a held
development city per rotation, with seeds 17, 29 and 43. Both forecast
rosters were frozen before outcomes were scored. The held role has 340
distinct scenarios, 1,020 seed views, and 465 views with complete labels.
These cities have already been used in the broader development program,
so this is not an independent confirmation result.

The release check keeps the original nonempty calibration prefix. A release
requires both the frozen position reference and candidate to pass the
unchanged CV/OLS4 gains and 2% risk rules, plus no conservative trajectory
loss and positive relative-trajectory gain. The same history-prefix score
orders both heads. Outside release, both policy arms use CV. This test
does not calibrate standalone deployment of the position reference.

## Held development outcomes

| Forced path | Qualifies | Strict gain over frozen position | Known ADE / available FDE | Observed signed easy harm >2% | Worst view p95 ADE |
|---|---:|---:|---:|---:|---:|
| Frozen soft cost | 89 | Not applicable | 0.250913 / 0.563983 | 171 | 7.390 |
| Frozen position projection | 119 | Reference | 0.248584 / 0.557725 | 152 | 7.367 |
| Individual refinement | 119 | 48 | 0.248574 / 0.557626 | 151 | 7.367 |
| Relative refinement | 119 | 44 | 0.248583 / 0.557685 | 152 | 7.367 |

Errors are in native coordinates. Frozen position improves qualification
over soft in each of the six cities. The city-weighted improvement is
3.685 percentage points (paired 95% CI [+1.337, +6.205]). Its direct
conservative trajectory gain over soft is +0.781 pp ([+0.079, +1.369]),
with better ADE/FDE and fewer easy exceedances. It still averages -0.862 pp
conservative gain against unedited CV. The direct conservative relative
gain versus soft remains negative: -0.00405 native units
([-0.00841, -0.00088]). The improved reference does not establish an
interaction gain.

| Relative refinement minus individual | Estimate | Paired city 95% CI |
|---|---:|---:|
| Relative gain on observed steps, same full-grid denominator (native units) | -0.0000089 | [-0.0000279, +0.0000131] |
| Missing relative disagreement debit (native units) | +0.0001308 | [+0.0000839, +0.0001860] |
| Conservative relative gain (native units) | -0.0001397 | [-0.0001900, -0.0000932] |
| Conservative individual trajectory gain (pp) | -0.01555 | [-0.02097, -0.01010] |

On fully labeled views, the relative contrast also does not establish an
improvement (paired 95% CI [-0.0000298, +0.0000083]). The conservative
failure is largely the missing-step disagreement, not a demonstrated gain
on known steps that was merely hidden by the metric. On calibration, one
nonempty prefix qualifies for individual and none for relative. Neither
arm releases a held view; there is no matched deployment gain.

Keep the position projection as a stronger **development reference**.
Do not deploy either residual head. The next useful question is whether
the unchanged position projection can beat CV and OLS4 at a nonzero,
supported release rate under the existing rules. Test this before training
another small angle head.

## Reproduction

36 new residual fits, 200 updates each, took 62.00 s and 845 MB peak RSS
on native arm64 CPU with four Torch threads and zero loader workers.
Checkpoints record both networks, optimizers and sampling RNG every
50 updates. The heartbeat gives PID, fold and progress.

The sealed scorer wrote all calibration and held receipts, but its final
JSON assembly reused the calibration key. The packer uses the same scoring
function and saves the hash under a distinct key. It reconstructs the
thresholds and verifies the old receipt before applying them. Without
`--score` it cannot open labels.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_relative_projection.py fit
.venv-pytorch/bin/python scripts/pack_m3w_av2_relative_projection.py --score
.venv-pytorch/bin/python scripts/pack_m3w_av2_relative_projection.py
.venv-pytorch/bin/python scripts/audit_m3w_av2_relative_projection_components.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

[Freeze](freeze.json), [calibration thresholds](calibration.json),
[scored summary](summary.json), [reference contrast](frozen_reference_contrast.json),
[loss attribution](contrast_attribution.json).
No official AV2 validation/test or DroneCrowd independent confirmation
data opened. No new CREATE job, manuscript, deployment, submission,
Stage5C or SMC change.
