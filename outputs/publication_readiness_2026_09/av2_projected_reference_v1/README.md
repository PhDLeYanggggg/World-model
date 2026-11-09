# AV2: learned directions improve qualification at the same edit budget

The [causal reference comparison](../av2_causal_reference_v1/README.md)
showed that selecting recipients differently did not yield a stronger
reference. Here I change the forecast directions while fixing both
recipients and the magnitude of their offsets from CV at every step.

## Design

The comparison keeps the same 8 observed and 12 predicted native steps,
CV and OLS4 floors, two recipients selected by the frozen additive
reference, validity masks, and original 2% conditions. For each step,
I project the predicted CV-relative displacement onto the circle of
the original radius. Neither the number of edits nor their radial
profile changes. Unedited tracks are unchanged. A zero directional
prediction keeps the original offset.

| Arm | Direction toward |
|---|---|
| Frozen additive | Original transported trajectory |
| OLS4 projection | Causal OLS4 position |
| Position-trained projection | A donor-trained position forecast |
| Risk-trained projection | The same model trained with additional projected ADE, dual-floor shortfall, and original risk excess losses |

The direction model is a small per-actor MLP using a local heading frame,
past positions, CV, OLS4 differences, and the radial profile. It does
not take the old edit direction as input. The two fits share architecture,
initialization and sampled options. Donor radii come from city-excluded
producers; donor training samples across all legal recipient assignments,
not recipients chosen by an in-sample teacher. Future labels are used
only in the donor loss and the calibration-role readout. The study
tests whether a better direction producer improves a *fixed-budget*
reference, not whether a new forecasting architecture beats
[EqMotion](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html)
on its benchmark.

Six rotations use four donor cities each and a separate calibration
city. Seeds 17, 29 and 43 give 36 fits, 400 updates per fit. Forecasts
were frozen before reading the 340 calibration scenarios (1,020 seed
views; 465 complete). No held-city outcomes for a rotation were read,
and the calibration comparison did not fit a threshold or release rule.

## Calibration readout

| Arm | Qualifies | Complete-label qualifies | Known ADE / available FDE | Worst view p95 ADE | Observed signed easy harm >2% |
|---|---:|---:|---:|---:|---:|
| Frozen additive | 65 | 48 | 0.25203 / 0.56531 | 7.432 | 184 |
| OLS4 projection | 32 | 24 | 0.25861 / 0.57295 | 7.500 | 218 |
| Position-trained projection | 105 | 79 | 0.24988 / 0.55871 | 7.432 | 154 |
| Risk-trained projection | 97 | 69 | 0.25383 / 0.56117 | 7.432 | 155 |
| CV (no edits) | Not an edited arm | Not applicable | 0.25184 / 0.56528 | 7.432 | Not applicable |
| OLS4 (unmatched budget) | Not an edited arm | Not applicable | 0.27987 / 0.59905 | 7.486 | Not comparable |

All error units are native. The position-trained arm gains 3.916
percentage points in city-weighted qualification versus frozen
additive (paired 95% CI [+1.601, +6.350]). The per-seed counts are
36, 37 and 32, against 22, 24 and 19 for the frozen additive arm.
Risk passes increase from 109 to 170, with the same 389 views having
a defined easy reference. The position-trained arm still averages
-1.578 pp in conservative gain against CV. Direct conservative
trajectory gain over additive is +0.880 pp ([-0.031, +1.734]).
It is not an established positive gain under missing-label debits.

Conservative relative-trajectory gain versus additive worsens by
0.00415 native units (95% CI for the gain [-0.00736, -0.00096]).
Both arms qualify and the position-trained arm improves relative
motion without individual trajectory degradation in 33 views.
That count is not a deployment policy. Extra risk losses do not
improve the position-trained arm on this readout. No replacement
is deployed.

## Exact geometry diagnosis

Allowing hindsight to point each step independently toward its
observed position, or toward OLS4 if missing, gives an exact relaxed
standalone ceiling at the same radii. The original additive recipients
permit 216 qualifying views (145 complete); allowing any recipients
raises that to 500. These numbers are not temporally smooth forecasts
or a learned model. The projection recovers some direction headroom
at fixed recipients, but does not solve recipient choice or the joint
trajectory objective.

Next, freeze the position-trained comparison and test direct
relative-trajectory supervision against the exact same producer and
budget, before evaluating the development holdout. Do not substitute
the larger qualification count for a matched interaction result.

## Reproduction

Native arm64 CPU, four Torch threads, zero loader workers. Training and
freeze took 46.51 s at 794 MB peak RSS. Checkpoints save both model
and optimizer states and sampling RNG every 50 updates; rerunning
`fit` resumes, and the heartbeat identifies the active fold and PID.
No additional CREATE job was needed. `replay` reads scalar receipts
without reopening labels; the capacity audit uses already scored
calibration outcomes.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_projected_reference.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_projected_reference.py score
.venv-pytorch/bin/python scripts/run_m3w_av2_projected_reference.py replay
.venv-pytorch/bin/python scripts/audit_m3w_av2_projected_reference_capacity.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

[Freeze](freeze.json), [fresh scoring](summary.json),
[direction capacity](direction_capacity.json). No AV2 official
validation/test, DroneCrowd independent confirmation, manuscript,
submission, deployment, Stage5C or SMC change.
