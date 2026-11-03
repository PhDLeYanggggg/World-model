# Same-Set Harm And Reference Diagnosis

## Material Passport

Fresh_run frozen-model cost inference on both SOURCE roles, not training.
All36cost heads are reused without updates;72role passes and a separate
complete replay finish. Held cost/amplitude arrays match the parent exactly.
This report reads public scalars; it is not another raw-data experiment.

## Same-Set Decomposition

Each value below is the median recording-view predicted/target mass ratio,
not prediction accuracy, a pooled pixel sum, or a ratio of population means.
H/EH predictions are amplitude-scaled; R/ER are not. Targets are actual
interpolated-path contributions: known masked costs plus the registered
wholly-unknown disagreement bound. They are not complete physical errors.

|Role|Arm|H ratio|R ratio|EH ratio|ER ratio|Completion all/easy violations|
|---|---|---:|---:|---:|---:|---:|
|critic_fit|known_only|0.459603|0.999594|0.004215|0.069396|170/364|
|critic_fit|bound_aware|0.331771|0.938287|0.006509|0.093521|203/358|
|double_excluded|known_only|0.407727|1.184316|0.004835|0.222316|177/413|
|double_excluded|bound_aware|0.392122|1.232014|0.039710|0.292533|174/332|

Every role/arm retains444recording views. Undefined reference ratios are
excluded from their own medians and reported as undefined in the JSON,
never scored as safe zero. These repeated views are not444independent sites.

## Selection And Fitting Failures

|Role|Arm|False-safe easy views|EH below target|ER above target|Selected full-EH ratio below full legal|
|---|---|---:|---:|---:|---:|
|critic_fit|known_only|364|364|35|310|
|critic_fit|bound_aware|358|358|40|423|
|double_excluded|known_only|413|413|104|319|
|double_excluded|bound_aware|332|332|124|411|

For bound-aware fitting views, median full-legal proposal EH ratio is
0.182912, falling to0.034390 on the actual selected set before amplitude.
This direction occurs in423/444views; the corresponding held count is411.
The same candidate target is used for this fixed-versus-selected comparison.
Actual interpolated EH ratios must not be substituted into that contrast.

All358fitting and332held falsely-safe bound-aware easy views underpredict EH.
ER is overpredicted in only40and124respectively. Thus denominator inflation
is not required for most of these failures. Numerator underestimation is
present even where denominator error is conservative. This component
measurement is stronger than attributing a ratio error to one term by guess.
It still does not identify a unique architectural or statistical cause.

The bounds can contribute to EH. A disclosed posthoc check retains the
full roster above but separately describes views with no unknown selected
rows: bound-aware fitting has16such views,13with defined EH, median EH
ratio0.003122 and8easy violations. Therefore missing-label bound mass is
not the sole explanation. This small selected slice is not a new primary
cohort or an independent test; all counts are retained in factor_counts.json.

## Implication For The Next Repair

Selected-cost fitting failure already exists before new-locality transfer.
The positive full-bound fitting change does not fix that conditional tail.
Do not repeat a width sweep, global attenuation grid or generic easy-event
hurdle and call it a new method. The earlier conditional-moment experiment
already improved occurrence/cost fit but lost policy utility. A useful next
candidate must explicitly address selected EH magnitude and wholly unknown
support contributions, with a distinct measurable change and matched controls.
Its selection rule, data roles and full-risk contract must remain prospective.

## Runtime And Reproduction

Original pass PID14300:197.365s,peak4,502,470,656bytes.
Separate complete replay PID14863:199.490s,peak4,522,606,592bytes.
All36job-role summaries reproduce exactly. Nativearm64CPU4/inter-op1/
workers0, heartbeat and per-job resumable outputs. Zero optimizer updates.

```sh
.venv-pytorch/bin/python scripts/run_m3w_cost_mass_decomposition.py register
.venv-pytorch/bin/python scripts/run_m3w_cost_mass_decomposition.py run
.venv-pytorch/bin/python scripts/run_m3w_cost_mass_decomposition.py verify
.venv-pytorch/bin/python scripts/report_m3w_cost_mass_decomposition.py
```

Completed phases are immutable; use run --resume only for an incomplete
original pass. Do not delete results to relabel cached work as fresh.

SOURCE is exposed development; fitting resubstitution is not independent
validation. Parent primary/risk failures, thresholds and deployment remain
unchanged. No calibration/confirmation/DroneCrowd arrays or new forecasts.
Pixels/raw frames/released silver are not metric, seconds or human gold.
Partial-mask ADE is not full12ADE. No physical-safety, true3D, foundation
or new dynamics claim. Stage5C and SMC remain off. Goal remains unmet.
