# When Frozen Neural Forecasts Add Feasible Choices

I kept CV on every untouched track and gave the same outcome-aware solver
either five analytic choices or those choices plus frozen EqMotion and PCT.
A witness counts only if reconstructing its forecasts beats both CV and OLS4
on the conservative full-grid score, and passes the original recording-level
all/easy 2% and support tests. Every cap is shared across menus; it does not
fix the realized number or amplitude of edits.

| Max edits/query | Analytic verified (/34) | Expanded verified (/102) | Confirmed new views (/102) | Distinct new recordings / sources |
|---:|---:|---:|---:|---:|
| 8 | 8 | 39 | 15 | 5 / 4 |
| 32 | 18 | 66 | 12 | 4 / 3 |

The analytic control is identical across the three frozen forecast seeds.
Confirmed new means the expanded witness verifies and the corresponding
analytic problem was ruled out, not merely unresolved. At cap 8, 1 analytic
and 12 expanded solves are unresolved; at cap 32, 5 and 21. The paired
source-macro increase in **verified** rate at cap 8 is +16.67 percentage
points (3,000 source-bootstrap draws: [+2.38, +34.52]). This interval is
descriptive for the verified results, not a test of actual deployment benefit.

## What the new witnesses use

- Cap 8: 60.15% of their edits use a neural direction; their median smaller
  gain against the two references is +11.00 points of known CV error mass.
- Cap 32: 67.16% of edits use a neural direction; the corresponding median
  is +14.73 points. Counts across seeds are repeated views, not independent
  recordings.
- I fixed the exact recipients and individual peak displacements, then
  let analytic directions shrink or extrapolate to those peaks. At cap 8
  only 1/15 still passes, while 13 are risk-infeasible and one has a negative
  upper bound. At cap 32 only 3/12 pass and nine are risk-infeasible. This
  post-hoc comparison rules out displacement alone as the explanation for
  most of these **particular** witnesses, not for every alternative plan.
- The new witnesses do not improve observed pair-displacement error over
  OLS4 in the changed-query diagnostic: -0.619 pixels at cap 8 and -2.851
  at cap 32. The conservative pair bound is negative even against CV.
  [Joint trajectory metrics](https://openaccess.thecvf.com/content/ICCV2023/papers/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.pdf)
  measure something different from the marginal gains and risk tests here.

The wider candidate set gives a joint decision model something to learn,
but this is not a causal selector. The next comparison should freeze the
same caps and candidate forecasts, use only source-excluded fitting data,
and compare a learned scene policy against independent per-agent gain
ranking. Keep a separate pair-geometry outcome: the observed marginal wins
do not show that the joint forecasts are better together.

## Reproduction

`fresh_run`: 272 local solver cells on the exposed 34 recordings from 14
sources, 306.43 s, maximum resident size 1.67 GB; post-hoc matched-shape
solves and pair diagnostic. `cached_verified`: original frozen, source-
excluded forecast arrays, and exact replay of stored choices and scores
without re-solving. 19 focused tests pass, including exhaustive tiny menus
and a fixed-shape rematch. `not_run`: training, external calibration,
independent confirmation, deployment, Stage5C and SMC. No change to the
2% rule, splits, or the 30 sealed DroneCrowd confirmation groups.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_cv_joint_feasibility.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_joint_feasibility.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_joint_witness_choices.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_joint_matched_shape.py
.venv-pytorch/bin/python scripts/build_m3w_cv_joint_replay.py
```

The first command resumes from completed solver cells. The anonymous ZIP
reproduces the aggregate checks, not raw trajectories or solver optimization;
its SHA-256 is `d1635f86494af08b3b8a72d86d11cda7ffbf30f65ba71b184dbaadf89844373d`.
The native eight-to-twelve annotation-step coordinates are local pixels,
not seconds, metric geometry, physical collision safety, or a foundation
world model result.

[Protocol](protocol.md) | [Readout](readout.json) |
[Same-recipient and same-peak diagnostic](matched_shape_diagnostic.json) |
[Manuscript v124](../evidence_manuscript_v124/README.md)
