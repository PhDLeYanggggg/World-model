# Frozen Interaction-Feature Cost Diagnosis

## Material Passport

Fresh_run frozen-model inference and target diagnosis on both SOURCE roles.
This readout reconstructs verified public scalars, not another training run.
All 36 models remain unchanged. H costs/amplitudes and completion-risk
ratios match the parent exactly; all 36 role groups replay exactly.

## Same-Selected-Set Cost Ratios

Median recording/pipeline-view predicted/target mass ratios are descriptive.
Each role/arm retains 444 repeated views, not 444 independent scenes.
H/EH predictions are amplitude-scaled; R/ER are not. Actual targets use
interpolated paths, observed labels and the original wholly-unknown bound.
Missing steps inside partially observed futures are not recovered.

| Role | Arm | H ratio | R ratio | EH ratio | ER ratio | All/easy violations |
|---|---|---:|---:|---:|---:|---:|
| critic_fit | zero64 | 0.374960 | 0.956183 | 0.013353 | 0.141783 | 175/348 |
| critic_fit | interaction64 | 0.331513 | 0.937621 | 0.003209 | 0.059937 | 182/324 |
| double_excluded | zero64 | 0.381852 | 1.249969 | 0.033021 | 0.343118 | 167/346 |
| double_excluded | interaction64 | 0.363251 | 1.209840 | 0.025886 | 0.250188 | 151/330 |

## Numerator And Denominator Errors

| Role | Arm | False-safe easy | EH below target | ER above target | Full legal EH ratio | Selected full EH ratio | Selected-ratio drop views |
|---|---|---:|---:|---:|---:|---:|---:|
| critic_fit | zero64 | 348 | 348 | 39 | 0.218875 | 0.053350 | 404 |
| critic_fit | interaction64 | 324 | 324 | 31 | 0.157515 | 0.027610 | 416 |
| double_excluded | zero64 | 346 | 345 | 121 | 0.252714 | 0.066978 | 409 |
| double_excluded | interaction64 | 330 | 330 | 92 | 0.222641 | 0.051748 | 419 |

The full-versus-selected comparison holds proposal targets fixed before
amplitude. It must not be replaced by the different actual-interpolation
ratio to attribute a selection effect. False-safe means predicted risk
<=2% and actual diagnostic risk >2% on exactly the same selected set.
This is a measurement of cost error, not proof of its unique cause.

## Missing-Label Sensitivity (Posthoc Descriptive)

| Role | Arm | Nonempty views with no unknown selected rows | Defined EH views | Median actual EH ratio | Easy violations |
|---|---|---:|---:|---:|---:|
| critic_fit | zero64 | 19 | 14 | 0.274722 | 9 |
| critic_fit | interaction64 | 19 | 15 | 0.021595 | 7 |
| double_excluded | zero64 | 4 | 4 | 1.577630 | 0 |
| double_excluded | interaction64 | 7 | 7 | 0.195662 | 2 |

These views do not replace the full roster, repair partial labels or
establish independent inference. Undefined reference ratios stay undefined.

## Execution And Reproduction

First pass PID 58823: 247.893 s, peak RSS 5,832,474,624 bytes.
Full replay PID 59410: 246.096 s, peak RSS 5,846,220,800 bytes.
Native arm64 Torch, CPU4/inter-op1, workers0. Zero optimizer updates.

```sh
.venv-pytorch/bin/python scripts/run_m3w_feature_cost_diagnosis.py register
.venv-pytorch/bin/python scripts/run_m3w_feature_cost_diagnosis.py run
.venv-pytorch/bin/python scripts/run_m3w_feature_cost_diagnosis.py verify
.venv-pytorch/bin/python scripts/report_m3w_feature_cost_diagnosis.py
```

Completed phases are immutable. Use run --resume only for an incomplete
same-definition pass; do not delete a result to relabel cached work fresh.

SOURCE remains exposed development. The parent primary and complete risk
remain failed/unsupported; this diagnosis changes no threshold, reference
or deployment. No calibration/confirmation/DroneCrowd input was opened.
Pixels/raw frames/silver are not metric, seconds or human gold. No new
neural dynamics, physical safety, true 3D or foundation claim. Stage5C and
SMC remain off. The research goal remains unmet.
