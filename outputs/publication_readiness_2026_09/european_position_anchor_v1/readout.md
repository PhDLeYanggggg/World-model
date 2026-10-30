# Causal position anchor: complete negative control

## Evidence Roles

fresh_run:24 full past-only forecast/error groups and a separate complete24-group replay.
cached_verified:original inputs,fit partitions,floor,physical predictions and producer43 identity.
not_run:no new neural training,independent calibration/confirmation,new RGB acquisition or deployment.
The separate error-geometry diagnosis is post-hoc,not another primary result.

Readout PID89466:21.532611s,peakRSS6105284608B.
Replay PID90008:20.764442s,peakRSS6074466304B.
All24 groups,24 row/support controls and48 original raw path/ADE/FDE controls agree exactly.
149,510 unique target rows,74 recordings,148 reused record/context views;12 exposed localities.
Unique row count is a private ID-set assertion in both full runs,bound by replay receipts;it is not inferred from public row sums.

## Registered Comparison

Hold OLS4/8 velocity fixed. Raw=current last observation+t*v;fitted=history mean+(k-1)/2*v+t*v.
Only the current-position anchor changes. Both use past observations only. Raw trajectories and costs reproduce originals exactly.
Keep the original causal eligibility and floor fallback,including each path exact-no-op exclusion.
This is not a matched-count selector. Counts happen to agree here;that is not the experimental constraint.
Primary:OLS8 fitted-minus-raw completion-lower utility. OLS4 is prespecified secondary,not a replacement winner.

|Fitted minus raw|Known utility pp [nominal95% CI]|Completion-lower pp [nominal95% CI]|
|---|---:|---:|
|OLS8|-19.349820 [-37.826305, -9.344567]|-21.078204 [-41.400334, -10.049531]|
|OLS4|-2.591366 [-4.793214, -1.391631]|-2.957059 [-5.420675, -1.614914]|

Both effects are negative in every one of the12 exposed localities. No locality is discarded.

|Candidate|Gated ADE gain over floor (%)|Ungated forecast ADE gain (%)|Full-roster FDE gain|Original risk support|Known all/easy violations|
|---|---:|---:|---|---:|---:|
|ols4_raw|-3.249440 [-11.975415, +2.668556]|-3.337737 [-12.156736, +2.692526]|not_estimable|0/148|148/148|
|ols4_fitted|-5.840806 [-17.002622, +1.266082]|-5.990298 [-17.051466, +1.223198]|not_estimable|0/148|148/148|
|ols8_raw|-28.319480 [-56.191070, -10.650670]|-28.644076 [-56.393925, -10.837850]|not_estimable|0/148|148/148|
|ols8_fitted|-47.669301 [-93.966486, -20.172115]|-48.337904 [-94.755230, -20.738174]|not_estimable|0/148|148/148|

Each candidate makes298,325 interventions over reused row/context views,including6,280 with wholly unknown futures.
All4 candidates have0 undefined easy-risk recording views;their failures are actual risk failures,not empty-control passes.
Original2% all/easy positive-harm/reference inequalities and easy-degradation requirements are unchanged.
Benefit does not cancel positive harm in the risk numerator. Gated here means original eligibility,not a calibrated safe selector.
Missing endpoints make the full-roster FDE aggregate not_estimable. Available endpoint errors remain in all group receipts and tail tables.

## Supplementary Hard/Easy Results

These are pooled known error masses over reused row/context views,not the locality-macro primary or independent samples.
|Candidate|Hard ADE gain (%)|Easy ADE gain (%)|
|---|---:|---:|
|ols4_raw|+8.397512|-34.481766|
|ols4_fitted|+7.918572|-42.063624|
|ols8_raw|+2.979026|-41.038534|
|ols8_fitted|-2.423326|-64.511153|

All148 recording/context tail views,each for4 candidates,are retained in [record_tail_table.csv](record_tail_table.csv).
All24 source/context pair-path proxies are in [pair_proxy_table.csv](pair_proxy_table.csv).
Tail p95 uses observed native local-image pixel errors within each record,not pooled physical safety.
Pair-distance contraction is a prediction-only,query-target proxy. It is not collision frequency,all-visible-agent coverage or physical safety.

## Fitting Checks

- complete24: pass
- all24_original_identity_controls_exact: pass
- all48_raw_path_and_cost_controls_exact: pass
- common_causal_eligibility: pass
- primary_lower_CI_positive: fail
- fitted_vs_floor_lower_CI_positive: fail
- complete_original_recording_risk: fail

4/7 checks pass. Advancement=False;deployment unchanged.

## Statistical And Claim Limits

Nominal3,000-draw paired locality bootstrap averages reused contexts within each of12 exposed SOURCE localities first.
Producer43 identifies the same fixed source forecaster;these deterministic predictions are not three new forecasting seeds.
This is not independent confirmation or whole-pipeline out-of-fold evaluation. Intervals are not corrected for the long adaptive research history.
All unknown labels and failed recording views remain. Completion bounds do not fill missing interior labels.
A failed blind fitted-position control does not prove every selectively chosen fitted prediction must fail.
It does reject this registered simple state-estimation repair on this fitting cohort. Raw anchoring was a modeling choice,not a demonstrated implementation bug.
Obs8/pred12/rawstride12,oblique detector-silver/local-image pixels,not SDDt50,metric,seconds,human gold,true3D or foundation success.
Independent calibration/confirmation,Stage5C and SMC remain closed. No stronger deployable model or submission-ready claim.

[Post-hoc mechanism](mechanism/readout.md),[failure analysis](failure_analysis.md),[figure](figures/position_anchor_result.svg),
[model/data card](model_data_card.md),[statistical review](statistical_review.md),[reproduction](reproduction_zh.md).
