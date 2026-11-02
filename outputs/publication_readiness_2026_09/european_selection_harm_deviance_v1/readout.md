# Matched H/EH Loss Training: Complete

Fresh_run:18complete Torch SOURCE cost fits, full first-producer refits,
18new frozen cost inferences and complete54group development readout/replay.
Cached_verified: mean_absolute control, forecast bank and SOURCE-only data/
normalization. No new forecaster, feature, goal, dataset or threshold search.
Selection is already exposed development; calibration arrays, confirmation
and DroneCrowd are not opened. Original failed experiments stay unchanged.

## Fixed Factors And Real Training

The easy-only arm replaces EH squared error with the existing stable cost
deviance. The dual arm replaces H and EH. B/R/ER and absolute gain anchoring
stay squared. Initialization,29,253parameters, SOURCE rows, sampler, draws,
optimizer and complete coverage match the original mean_absolute control.
All three50->100recovery checks and the original100step control are exact;
the complete original control refit is also exact. Both new first-producer
full refits match. These are reproducibility checks, not generalization.

|Arm|Fits|Primary updates|Mean first minibatch total|Mean last minibatch total|Mean last common MSE/gain objective|
|---|---:|---:|---:|---:|---:|
|easy_deviance|9|70656|3.320995|0.249222|0.253766|
|dual_deviance|9|70656|3.257773|0.298546|0.291336|

Loss definitions differ; raw totals across arms are not a common test
metric. First/last minibatches differ. The original-total monitor is the
same fitting loss, not validation, and does not establish downstream lift.

## Complete Development Policies

|Policy|Known ADE gain % [95% CI]|Supported views|Known all/easy violations|Undefined easy|Worst signed easy net %|
|---|---:|---:|---:|---:|---:|
|mean_absolute|+1.503747 [+0.816987, +2.130741]|6/252|199/211|14|8.49547|
|easy_deviance|+0.091837 [+0.008577, +0.237160]|33/252|139/127|67|2.52267|
|dual_deviance|+0.086937 [+0.015525, +0.179341]|30/252|134/133|64|1.62226|

All38,102targets/28recordings/six localities and nine pipelines remain.
The252risk views repeat recordings and are not independent samples. Unknown
and partial outcomes are retained. Original2%all/easy positive-harm and
easy-net checks apply to every required view; undefined support is not safe0.

## Registered Contrasts

|Arm minus mean_absolute|Known pp [95% CI]|Direct paired completion-lower pp [95% CI]|
|---|---:|---:|
|easy_deviance|-1.411910 [-2.030263, -0.748412]|-1.944856 [-2.608577, -1.249594]|
|dual_deviance|-1.416810 [-2.059565, -0.709034]|-1.977437 [-2.656544, -1.259025]|

Registered primary is the dual arm FULL paired completion-lower effect.
Primary pass: **False**. Original record-risk pass:
**False**. Candidate: **False**.
No alternate arm or favorable mean replaces a failed registered primary.
Intervals use3,000locality resamples, averaging repeated folds/actual training
seeds17/29/43 within each site. These are conditional development intervals.
Unknown-pair disagreement is computed directly, not as two lower bounds
subtracted. No population-risk guarantee or independent confirmation follows.

## SOURCE Cost Diagnosis

|Arm|Pooled selected H predicted/true|Pooled selected EH predicted/true|Known selected all/easy risks %|
|---|---:|---:|---:|
|easy_deviance|0.280146|0.20509|6.24669/7.22475|
|dual_deviance|0.473251|0.214037|3.69465/5.56856|

These pool repeated SOURCE fitting action views, not held-out validation.
Selected pools differ by arm; their prediction ratios are not fixed-set error reductions.
Improved fit or cost ratios cannot substitute for the development policy and
original risk checks. The old144-fit easy-deviance failure is distinct and
remains negative; this adaptation is not a new divergence contribution.

### Fixed All-Legal Bank: Common Normalized Squared Error

|Arm|Observed action views|H MSE|EH MSE|R MSE|ER MSE|
|---|---:|---:|---:|---:|---:|
|mean_absolute|5267952|0.0034707909|0.0004150098|0.088695336|0.0019800555|
|easy_deviance|5267952|0.0032932242|0.00049331054|0.089040054|0.0020617051|
|dual_deviance|5267952|0.0031514197|0.00048681178|0.090143794|0.002130166|

Counts and observed target masses match across arms; normalization and bank stay fixed.
These repeated fitting actions are not independent held-out data. Dual H MSE decreases,
but EH MSE increases; a better selected-pool ratio is not blanket cost accuracy.

## Runtime And Actual Resource Recovery

|Phase|PID|Seconds|Peak RSS bytes|
|---|---:|---:|---:|
|pilot|30702|8.459041|8536358912|
|training_freeze|32164|448.537613|12097355776|
|policy_freeze|33416|26.771173|2631122944|
|complete|33725|19.469587|2789982208|
|verification|34105|29.533293|2759868416|
|resource_resume_verification|33193|39.189826|10574397440|

The first training process hit the10GiB disk-reserve guard after six fits
and checkpoint2000/8192of single1_seed17/easy_deviance. Free space recovered
without deleting any data/cache/system file; the fluctuation cause is not
established. The unchanged run resumed. Training-freeze runtime covers the
completion process only, not the earlier interrupted process. Per-fit traces
and optimizer time are retained. A separate direct8192update refit matches
the actually resumed full model/optimizer/sampler/draws/trace/RNG exactly.
No budget/guard reduction. CPU4/inter-op1/workers0, native arm64.
CREATE queue access was verified read-only; no remote jobs submitted/changed.

## Reproduction And Boundaries

```sh
.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py pilot
.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py infer --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py eval --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py verify
.venv-pytorch/bin/python scripts/verify_m3w_harm_training_resource_resume.py
.venv-pytorch/bin/python scripts/report_m3w_selection_harm_deviance.py
```

Completed phases are immutable; do not delete them to rerun. Unchanged partial
training resumes from its full atomic checkpoints. Private SOURCE files and
weights are required for real execution; this is not a cold-start release.
No learned deployment changes. Available-label ADE is not full12 ADE; bounds
cover wholly unknown rows, not missing steps inside partial futures. Raw
stride12/local pixels/silver are not seconds/meters/human gold. No true3D,
foundation or new trajectory-dynamics claim. Stage5C/SMC remain off.
