# Online selected-action loss: complete readout

## Material Passport

Mode: experiment execution and verification. Data role: exposed SOURCE fitting only.
Result source: fresh native arm64 Torch training and inference; cached verified inputs and trajectory banks.
Status: complete and separately replayed exactly. Independent confirmation: not run.

## What changed

I added one unit-weight loss on the currently chosen supervised false-admitted action.
The full upper-score/five-moment/ranking loss remains. Inputs, width, initialization,
sampling, normalization, proposal bank, decoder and original 2% risk rule do not change.
This is a cost-head learning experiment, not a newly trained trajectory producer.

All 54 heads complete 2,000 updates: 108,000 primary updates, 2,000 full-refit updates and 160 pilot updates.
All 36 original/upper full scientific states match; the admitted head full refit is exact.
Training PID 96918 took 378.320s, peak RSS 10893541376B.
Readout PID 99574 took 120.873s, peak RSS 10917429248B.
All 72 groups and 144 control prediction/action/full-metric checks agree in a separate replay.
Private primary checkpoint size: 23064998B. All 54 actual traces accompany this report.

## Complete-policy results

Known ADE gain (%) versus the frozen floor; equal locality, 95% bootstrap interval.
Three controller seeds, 3,000 resamples over 12 localities after context/seed averaging.

|Objective|Known ADE gain|Supported recording views|Known easy risk violations|Known all risk violations|Undefined easy views|
|---|---:|---:|---:|---:|---:|
|original|+1.893446 [+0.692104, +3.721837]|37/444|313|179|32|
|upper_score|+1.979966 [+0.655820, +3.950357]|41/444|312|156|42|
|admitted_score|+1.904562 [+0.660504, +3.710170]|46/444|308|149|42|

## Fixed contrasts

|Contrast|Known pp [95% CI]|Completion-lower pp [95% CI]|
|---|---:|---:|
|admitted_minus_upper|-0.075404 [-0.245661, +0.026133]|-0.120265 [-0.290321, -0.012139]|
|admitted_minus_original|+0.011116 [-0.234681, +0.327104]|-0.860081 [-2.088728, -0.059617]|
|Same admitted head actual neural use|+0.194407 [+0.049385, +0.386839]|+0.164782 [+0.019557, +0.359793]|

Completion sensitivity varies wholly unknown-row policy disagreement only. Partial observed
ADE remains fixed at its original available-point estimand; it is not full 12-step recovery.
The complete-roster FDE is not estimable where a required locality lacks endpoints.
Same-count comparisons are utility-only; empty is not estimable, never safe zero.

## Selection failure mechanism

|Objective|Selected known|Selected unknown|Easy false admissions|Easy numerator-only|Observed easy harm|Predicted easy harm|
|---|---:|---:|---:|---:|---:|---:|
|original|263581|3511|39562|38666|24672.550435|281.923647|
|upper_score|262370|3658|33241|32532|19341.876351|255.500950|
|admitted_score|262035|3636|33143|32446|19255.862315|254.215456|

Arithmetic witnesses use future evaluation labels only, never deployment rules or causal proof.
Pooled counts and masses are overlapping fitting views, not independent samples or metric quantities.
Support remains 149,510 unique queries, 74 recordings, 12 localities; no test/independent outcome opens.

## Gates

- complete54heads72groups: pass
- exact36parent_controls: pass
- complete_recording_risk: fail
- primary_known_CI_positive: fail
- primary_completion_lower_CI_positive: fail
- actual_neural_use_completion_lower_CI_positive: pass

Fitting-progress gates: 3/6. Complete advancement: False.
No independent confirmation, deployment change or submission-ready claim follows from fitting gates.

## Limitations and next action

The original 2% selected-reference rule remains mandatory in every required recording. Positive
utility, fewer unsafe rows or positive actual-neural-use contrast cannot replace that rule.
This experiment uses exposed development data; all intervals are conditional and exploratory.
If recording risk or primary utility still fails, retain the fallback and diagnose the changed
selected population before another distinct repair. Do not tune on reserved outcomes.
Independent calibration/confirmation and a defensible primary contribution remain required.
This supplied detector-silver top-down task remains pixel/raw-frame, not metric, seconds-level,
human gold, true 3D or foundation world modeling. No Stage5C execution or SMC.
