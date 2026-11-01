# Coupled benefit/risk: complete SOURCE development result

## Material Passport

fresh_run:36 real Torch cost-head fits,72,000 primary updates,two exact full refits,and complete72-group policy readout/replay.
cached_verified:original past inputs,forecasts,floor,normalizers and old protected full-policy choices.
not_run:new trajectory/dynamics training,independent calibration/confirmation,SDD deployment retest.
All72 protected model predictions and full-joint policy controls are exact. Controls are not relabeled independent tests.

## Registered Factor

Release only the benefit constraint in the selected-set risk recalibrator. Both models have30,857 parameters,
identical initialization/features/draws and2,000 updates. Protected updates reproduce the old model exactly.
Both optimize0.5*(benefit_MSE+pointwise_risk+selected_query_score);benefit_MSE is constant for the protected arm.
Original reference predictions R/easy_R,all observed evaluation targets,strict2% positive-harm/reference budgets and easy preservation remain unchanged.
Primary:released minus protected FULL-joint completion-lower utility. Matched-count is a secondary utility control,not a safety replacement.

|Released minus protected|Known pp [95% CI]|Completion-lower pp [95% CI]|
|---|---:|---:|
|joint|-0.543227 [-0.820685,-0.301313]|-11.080186 [-28.151170,-2.019377]|
|matched|-0.225892 [-0.397646,-0.061245]|-4.043770 [-8.837262,-1.224683]|

|Policy|Known ADE gain over floor (%)|Completion-lower pp over floor|Full-roster FDE|Supported original record-risk views|Known easy-risk violations|Undefined easy|Selected / unknown|
|---|---:|---:|---|---:|---:|---:|---:|
|protected_independent|+6.866192 [+4.048840,+10.528377]|-6.303657 [-27.839429,+6.601086]|not_estimable|1/444|349|13|410575/7628|
|protected_joint|+8.215347 [+4.831265,+12.277071]|-5.568909 [-27.273581,+7.677735]|not_estimable|1/444|383|13|410575/8430|
|protected_matched|+7.608717 [+4.341526,+11.544053]|+0.394679 [-8.798763,+7.321842]|not_estimable|2/444|372|19|354564/7446|
|released_independent|+6.211251 [+3.638696,+9.573910]|+1.995311 [-3.121374,+6.649651]|not_estimable|8/444|330|16|378174/6659|
|released_joint|+7.672120 [+4.475445,+11.511371]|+3.196884 [-2.265523,+8.293335]|not_estimable|6/444|362|16|378174/7256|
|released_matched|+7.382825 [+4.153665,+11.209493]|+3.111411 [-2.059673,+8.107455]|not_estimable|4/444|366|19|354564/6913|
|floor|+0.000000 [+0.000000,+0.000000]|+0.000000 [+0.000000,+0.000000]|not_estimable|0/444|0|444|0/0|

Genuine positive-harm/reference risk is not net error;benefit never cancels its numerator.
The released floor-relative lower point is +3.196884 [-2.265523,+8.293335] pp;its interval still crosses zero.
Subtracting two individual lower-bound points does not give the paired lower bound. It cannot replace the negative registered paired effect.
All149,510 unique target rows,74 recordings,12 exposed localities and444 reused record/context/critic-seed views remain.
Wholly unknown outcomes remain in bounds and are not safe zeros. Empty matched risk is not_estimable.
The floor makes no intervention;its selected-risk denominator is absent,not evidence that the baseline adds harm.
Unavailable required endpoints keep full-roster FDE not_estimable;available errors remain supplementary.

## Fitting Checks

- complete36fits72groups: pass
- all72_old_prediction_and_full_policy_controls_exact: pass
- matched_features_initialization_and_draws: pass
- same_count_utility_controls: pass
- primary_released_minus_protected_lower_CI_positive: fail
- released_floor_lower_CI_positive: fail
- complete_original_recording_risk: fail

Advancement:False. Deployment unchanged,even if an exploratory fitting check passes.

## Limits

Intervals are nominal3,000-draw locality bootstraps after averaging reused contexts and critic seeds within12 EXPOSED SOURCE localities.
Not independent confirmation,not whole-pipeline OOF,and not adjusted for adaptive prior research. Three critic seeds share forecast producer43.
Any paired cost-head effect includes changes to predicted utility and admissible harm capacity;it is not isolated proof of physical dynamics.
Record tails,hard/easy pooled known masses,and prediction-only pair-distance contraction are in complete aggregate group receipts.
Pair proxies cover query targets,not all visible agents,and do not certify physical collision safety.
A lower fitting monitor does not establish decision calibration,conformal control or deployment.
Obs8/pred12/rawstride12,oblique detector-silver/local-image pixels,not SDDt50,metric,seconds,human gold,true3D or foundation.
No independent role,Stage5C execution or SMC. Not submission-ready.

[Policy figure](figures/policy_result.svg),[training curves](training_report.md),[method and primary-source compatibility](method_note.md),
[frozen protocol](protocol.md),[failure analysis](failure_analysis.md),[reproduction](reproduction_zh.md).
