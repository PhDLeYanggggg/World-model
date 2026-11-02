# Current-Bank Nonlinear Cost Control: Complete

## Material Passport

Fresh_run:18matched ExtraTrees SOURCE fits, one complete primary-arm refit,
18development cost/action inferences,54metric groups and full separate replay.
Cached_verified: original mean_absolute neural cost head, frozen forecasters,
causal inputs, normalization, complete query-draw multiplicities and labels.
No new neural/trajectory training, threshold search or independent-role opening.

## Why This Comparison Is Matched

All nine current producers use SOURCE folds and actual training seeds17/29/43.
Each cost head trains on the original eight producer-excluded SOURCE localities.
Both forests use444causal fields, action identity and causal envelope; the
context arm adds fixed leave-self-out same-query means. Original query visits,
legal-action/agent weighting and absolute-gain quadratic objective are preserved.
The fixed forest context and post-fit physical projection differ from neural
learned-hidden pooling and coupled decoding. This is not a one-factor capacity
proof or matched-FLOP benchmark. The old forest experiments remain distinct.

## Complete Development Policies

|Estimator|Known ADE gain % [95% CI]|Mean switch %|Supported views|Known all/easy violations|Undefined easy|Worst signed easy net %|
|---|---:|---:|---:|---:|---:|---:|
|mean_absolute|+1.503747 [+0.816987, +2.130741]|22.495891|6/252|199/211|14|8.49547|
|row_forest|+0.057865 [-0.000173, +0.119967]|0.681326|67/252|33/41|143|0.102127|
|context_forest|+0.048919 [-0.000077, +0.101528]|0.534815|56/252|34/37|158|0.617861|

All38,102targets/28recordings/six exposed development localities remain.
The252pipeline/recording risk views repeat recordings, not independent samples.
Original2%all/easy positive-harm, easy-net and finite-support rules are unchanged.
Empty selected easy reference is not a safe zero. Partial/unknown labels remain.

## Registered Contrast

|Forest minus original mean_absolute|Known pp [95% CI]|Direct paired completion-lower pp [95% CI]|
|---|---:|---:|
|row_forest|-1.445882 [-2.092086, -0.725490]|-1.998658 [-2.683745, -1.276683]|
|context_forest|-1.454827 [-2.093394, -0.727516]|-2.010571 [-2.703014, -1.292373]|

Primary:context_forest FULL direct paired completion-lower contrast, not a
selected best arm or the subtraction of two individual lower bounds.
Primary pass:**False**. Complete original risk pass:**False**.
Development candidate criterion:**False**. Deployment remains unchanged.
Intervals use3,000locality resamples after source-fold/actual-seed averaging.
These are conditional exposed-development intervals, not final confirmation.

## Fixed All-Legal SOURCE Bank

|Estimator|Known action views|H normalized MSE|EH normalized MSE|R normalized MSE|ER normalized MSE|
|---|---:|---:|---:|---:|---:|
|mean_absolute|5267952|0.00347079|0.00041501|0.0886953|0.00198006|
|row_forest|5267952|0.00307054|0.000427477|0.0727874|0.00200762|
|context_forest|5267952|0.00307414|0.000427953|0.0732025|0.00200302|

All-legal observed action counts and cost masses match. These squared errors
use the frozen causal scale, uniformly pooled action views; they are not the
query-weighted/RMS training objective or a held-out conditional-risk estimate.

### Complete Decoded SOURCE Objective

|Estimator|Full objective|Gain MSE|B MSE|H MSE|R MSE|ER MSE|EH MSE|
|---|---:|---:|---:|---:|---:|---:|---:|
|mean_absolute|0.7215|0.382875|0.428758|0.247987|1.24971|0.320995|0.402858|
|row_forest|0.918171|0.54788|0.566015|0.327095|1.05767|0.505625|0.764752|
|context_forest|0.920199|0.549883|0.566067|0.328166|1.06478|0.505895|0.761374|

These are complete decoded query-visit/RMS-weighted objectives against the
original float32 training targets,averaged across nine producers. Wholly
unlabeled queries retain zero supervision. They are not the fixed subset
unprojected forest monitor or uniformly pooled error above. In this metric
both forests fit worse,especially EH and net gain. This finite control does
not establish that stronger regression solved SOURCE fitting,or that the
causal inputs are irreducibly insufficient. Decode/reference-sharing effects
remain to be separated before another model or loss change.

### Selected SOURCE Regions

|Estimator|Selected known actions|H predicted/true|EH predicted/true|Known all/easy risks %|
|---|---:|---:|---:|---:|
|mean_absolute|235222|0.281149|0.00364461|5.10903/11.4495|
|row_forest|2669|6.48307|3.28771|0.192636/0.751589|
|context_forest|1903|6.40963|2.9423|0.203548/0.798263|

Selected pools differ; prediction-ratio changes cannot be treated as fixed-set
error improvements. SOURCE fitting resubstitution is not generalization.

## Compute And Recovery

|Phase|PID|Seconds|Peak RSS bytes|
|---|---:|---:|---:|
|pilot|37889|214.392408|9218342912|
|training_freeze|50966|3670.010830|15547973632|
|policy_freeze|57645|50.096892|4892819456|
|complete|57841|17.153881|2624372736|
|verification|57985|211.200816|16272457728|

Training resumed after a turn interruption from the actual tenth model
96/128tree checkpoint. The original process was absent; its exit code and
exit mechanism are unknown,not evidence of an OpenMP/SHM hang. Complete
previous models were revalidated,not refitted. Training-freeze runtime
covers only the resumed process; per-fit retained work is recorded in
the fit receipts. Unsaved interrupted-block work is unknown. No budget
or resource guard changed.

The actual interrupted96/128tree producer also matches an independent
direct128tree refit,including all retained96prefix trees and trace.
This adds128reproduction trees,distinct from the first-producer refit.

The complete primary budget is2,304trees plus128first-primary refit trees.
Native arm64, four fit threads, single-thread predictions, workers0; no
DataLoader multiprocessing. Atomic checkpoints and the10GiB reserve remain.
Pilot8->16/direct16recovery and full128tree primary refit are exact.
Full18SOURCE and18development cost inferences plus54metric groups replay.

## Reproduction And Limits

```sh
.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py pilot
.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py infer --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py eval --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py verify
.venv-pytorch/bin/python -m scripts.report_m3w_selection_cost_forest
```

Completed phases are immutable. Private SOURCE/model assets are required;
the public code is not a complete cold-start anonymous reproduction package.
Available-label ADE is not full12ADE. Unknown-row bounds omit missing steps
inside partial futures. Local-image pixels/raw stride12/silver are not metric,
seconds, human gold or physical safety. No true3D/foundation or submission-
ready claim. Calibration private arrays, confirmation and DroneCrowd remain
closed in this study; the older calibration exposure remains disclosed.
Stage5C and SMC remain off.
