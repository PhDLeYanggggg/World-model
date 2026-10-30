# Capped norm-parity admission learning

## Material Passport

Fresh run: 18 newly trained cost heads and fresh inference. Cached verified: 36 frozen upper/unit control models.
All data are exposed SOURCE fitting/resubstitution. Independent confirmation, deployment and submission readiness remain false.

## Intervention and actual execution

The preceding exact training replay found frequent auxiliary activation but a median
parameter-gradient ratio of only 0.003632%. This trial changes only its detached scale:
`lambda=min(1e6, ||grad anchor||/||grad auxiliary||)` when defined; otherwise zero.
The same five-moment/ranking/upper anchor, auxiliary, model, features, normalization,
sample draws, candidate bank and 2% inference rule remain. This is not full GradNorm,
a new deferral framework, moment calibration or a safety theorem.

All 18 fresh heads complete 2,000 steps: 36,000 primary updates, 2,000 exact-refit updates and 400 pilot updates.
Training PID 9772 takes 209.663s, peak RSS 10916429824B.
All 36 frozen controls match common initialization/preprocessing/RNG/draw design; they are not claimed as newly trained.
Readout PID 12163 takes 125.407s. A separate process repeats all 72 groups and the summary exactly.
Both controls reproduce all 144 prediction/action/full-policy metric checks.

## Did the optimization mechanism operate?

Across 36000 fresh steps: 28464 zero-scale and 6163 capped steps.
At 163/738 monitored snapshots the scale is positive.
Nonzero achieved auxiliary/anchor gradient-ratio quantiles (min,p10,median,p90,max): [0.003409396786777881, 0.012980267943201895, 0.13231752648552386, 1.0, 1.0000000000000002].
These are pre-clipping gradient ratios, not AdamW update ratios, calibrated risks or independent samples.
The numerical cap is retained and disclosed, not adjusted after scores.

## Complete-policy accuracy and risk

Known ADE gain (%) versus the frozen floor, equal locality with 95% bootstrap intervals.
Three controller seeds and 3,000 locality resamples after context/seed averaging.

|Objective|Known ADE gain|Supported recording views|Known easy violations|Known all violations|Undefined easy views|
|---|---:|---:|---:|---:|---:|
|upper_score|+1.979966 [+0.655820, +3.950357]|41/444|312|156|42|
|admitted_score|+1.904562 [+0.660504, +3.710170]|46/444|308|149|42|
|balanced_score|+0.388168 [+0.089387, +0.857230]|201/444|31|30|72|

|Fixed contrast|Known pp [95% CI]|Wholly-unknown completion-lower pp [95% CI]|
|---|---:|---:|
|balanced_minus_admitted|-1.516394 [-2.869731, -0.491213]|-2.101592 [-3.424839, -1.044353]|
|balanced_minus_upper|-1.591798 [-3.113312, -0.481300]|-2.167074 [-3.651269, -1.019025]|
|Same balanced head actual neural use|+0.034936 [+0.006029, +0.084052]|+0.030673 [+0.003886, +0.079485]|

A positive secondary or known-only mean cannot replace complete-policy risk or the fixed primary.
Same-count controls are utility-only; empty is not estimable. The complete-roster FDE is
not estimable when a required view lacks endpoints. Completion sensitivity bounds wholly
unknown rows only, not all missing positions inside partially observed ADE labels.

## Selected easy-harm diagnosis

|Objective|Selected known|Selected unknown|Easy false admissions|Observed easy harm|Predicted easy harm|
|---|---:|---:|---:|---:|---:|
|upper_score|262370|3658|33241|19341.876351|255.500950|
|admitted_score|262035|3636|33143|19255.862315|254.215456|
|balanced_score|25364|464|2453|237.546519|17.979441|

Arithmetic witnesses use evaluation labels only and are not inference-time policies or causal identification.
Counts/masses are overlapping fitting views, not independent new data. The same 149,510 unique
queries, 74 recordings and 12 localities remain; unknown outcomes are not imputed.

## Remaining risk and support failures

Failure counts (overlapping, not a partition): {'condition_failed_all_selected_risk_upper': 171, 'condition_failed_easy_selected_risk_upper': 238, 'all_completion_upper_fails_despite_known_budget': 94, 'easy_completion_upper_fails_despite_known_budget': 135, 'condition_failed_easy_degradation_upper': 76, 'easy_undefined': 72, 'all_undefined': 47}.
The same frozen bank has cached label-aware positive feasible opportunity in all 148
seed-deduplicated context/record views. This uses evaluation labels and whole-record
allocation; it does not show inference-time identifiability or simultaneous-frame feasibility.
It is cached verified evidence, not a fresh oracle run or a learned policy result.

## Fixed fitting gates

- complete18fresh36cached_endpoints72groups: pass
- cached_control_inference_and_scores_exact: pass
- complete_recording_risk: fail
- primary_known_CI_positive: fail
- primary_completion_lower_CI_positive: fail
- actual_neural_use_completion_lower_CI_positive: pass

Progress gates: 3/6; advancement: False.
Even passing fitting gates would not establish independent generalization or submission readiness.

## Scope and reproduction

Native arm64 CPU4/interop1/workers0; checkpoint100/heartbeat50, explicit resume and bounded runtime.
All 18 actual loss traces and balance statistics are retained; raw inputs, caches and weights remain private.
This is supplied detector-silver top-down pixel/raw-frame forecasting, with eight past and twelve requested
steps at raw stride12. It is not the supplemental raw-frame t+50 task, metric, seconds-level, human gold,
true3D or a foundation model. No Stage5C execution or SMC. No new scene-joint dynamics claim.
