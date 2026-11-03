# Public EqMotion Fixed-K1 SOURCE Comparison

## What Was Run

Fresh_run: nine full native-arm64 Torch EqMotion fits, a complete first-model
refit, actual 50->100/direct-100 recovery pilot and complete SOURCE-excluded
inference/replay. Cached_verified: original PCT fits, source roles and arrays.
Every PCT control forecast was also rerun from its original checkpoint.

The original forecaster cohort contains 318,969 indexed rows at twelve SOURCE
localities. This is not the later 149,510-row controller whitelist or the six
exposed development localities. Each locality has two excluded producer folds
and actual training seeds 17/29/43. Overlapping windows and repeated producer
views are not independent scenes. SOURCE crossfit is not final confirmation.

## Fixed Adaptation

Official implementation: [EqMotion](https://github.com/MediaBrain-SJTU/EqMotion),
MIT, pinned commit `5aec2e0b61c511fa93a24138dd90da59a089084b`. All eight pinned
source files were verified before execution. Fixed head 0 emits one deterministic
trajectory; no label-based best-of-20 choice, pretrained weights or author trainer.
The common input conditioner uses past ego/neighbor geometry only and restores
output units before loss/evaluation. Observation 8, prediction 12, raw stride 12.
The original PCT has a motion-bounded output wrapper; EqMotion emits coordinates.
PCT also consumes its causal baseline query, while the EqMotion core ignores it.
This is a family comparison, not isolated equivariance, equal compute or published
best-of-20 reproduction. No new selector or policy was fitted or deployed.

## Complete Forecast Results

|Model|Equal-locality available-label ADE gain vs CV %|
|---|---:|
|eqmotion|+17.213795|
|PCT|+2.690215|
|constant_position|-87.361789|
|constant_velocity_causal_fd|+0.000000|
|damped_velocity_090|+1.086996|
|damped_velocity_097|+3.975483|
|history_ols4_velocity|+3.535671|
|history_ols8_velocity|-9.468883|
|Fit-selected causal baseline|-0.838779|

The fit-selected baseline uses its own four fitting localities only; it
need not be strongest on the excluded SOURCE groups. Fixed-baseline readout
rankings do not retrospectively change the floor.
Planned known-label EqMotion-minus-fit-selected contrast pp: +18.052574 [+12.196434, +28.163065].

|EqMotion minus PCT|Paired effect pp [95% locality CI]|
|---|---:|
|known_pp|+14.523580 [+9.550578, +22.186110]|
|completion_lower_pp|+5.787942 [-0.931986, +14.733543]|
|completion_upper_pp|+23.259219 [+17.858427, +30.357053]|

Registered known-ADE primary positive: **True**.
Intervals use 3,000 locality resamples after averaging fold/actual-seed effects.
Completion bounds cover wholly unknown rows only; they do not impute unobserved
steps inside partial futures or identify full-12-step ADE. They share a fixed known
CV-error denominator, not an inferred full-population error mass. Endpoint FDE
is available only where the twelfth requested endpoint is actually observed.

## Locality Breakdown

|SOURCE locality|Known difference pp|Completion-lower pp|Completion-upper pp|
|---|---:|---:|---:|
|eu-locality-007|+7.144623|+3.863460|+10.425787|
|eu-locality-008|+20.424922|+12.061109|+28.788735|
|eu-locality-020|+50.180862|+44.952938|+55.408787|
|eu-locality-048|+13.835346|+0.371721|+27.298971|
|eu-locality-067|+8.138894|-0.422924|+16.700712|
|eu-locality-074|+9.184204|+4.208603|+14.159806|
|eu-locality-082|+6.940288|-13.872464|+27.753041|
|eu-locality-110|+15.620295|+12.917387|+18.323202|
|eu-locality-112|+15.443254|+8.836978|+22.049531|
|eu-locality-119|+6.990182|-0.732629|+14.712992|
|eu-locality-124|+11.439574|-2.342151|+25.221299|
|eu-locality-126|+8.940518|-0.386725|+18.267761|

## Easy, Hard And Endpoint Support

|Subset|Missing required producer/locality views|PCT gain vs CV %|EqMotion gain vs CV %|
|---|---:|---:|---:|
|easy|0|-13.226759|+0.915299|
|hard|0|+9.440589|+18.366945|
|complete|0|+5.601001|+22.759205|

Missing required subgroup support remains N/A, not a safe zero or an excuse
to silently drop views. Per-group available ADE, endpoint FDE, label coverage,
zero-CV harms, all six baselines and fit-selected strongest baseline are retained
in the 72 public scalar group receipts. No physical or complete-policy risk
certificate is inferred from this forecasting comparison.

## Compute And Exact Reproduction

All nine models store 3,027,268 parameters, with 580,676 gradient-enabled
parameters and 386,628 parameters that actually have AdamW state at step 4,000.
These counts are not interchangeable: frozen unused heads remain stored, and
some gradient-enabled core paths receive no gradient under this fixed forward.
That accounting alone identifies no unique performance failure cause.

Primary updates: 36000; full refit: 4,000; pilot: 200.
Full training/refit process: PID 66505, 12893.020 s,
peak RSS 2,251,620,352 bytes. Separate inference replay:
PID 92338, 7546.152 s, peak 1,923,252,224 bytes.
Original query draws, factors, fitting IDs and held-SOURCE exclusions match.
All nine new and nine old model forecasts and all 72 metric groups replay.
Native arm64, CPU4/inter-op1/workers0, atomic checkpoints and heartbeat.
Real recovery and a full refit are reproducibility checks, not independent evidence.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py pilot --resume
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py infer --resume
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py eval
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py verify
.venv-pytorch/bin/python scripts/report_m3w_european_eqmotion.py
```

Completed stage receipts are immutable. Resume incomplete stages only; do not
delete a complete receipt to manufacture a fresh run. Private source arrays and
checkpoints are still required, so anonymous cold-start reproduction is not complete.

## Verdict And Limits

No development, calibration, confirmation or DroneCrowd arrays were opened.
The historical calibration remains completed and failed, not untouched data.
No current deployment changes; the original full 2% recording-risk contract
was not retuned or replaced by SOURCE forecast accuracy. This result is not a
JEPA contribution, world-dynamics upgrade or submission-readiness certificate.
Image-local pixels and detector/tracker silver are not metric, seconds, human
gold, physical safety, true 3D or foundation evidence. Stage5C and SMC remain off.
