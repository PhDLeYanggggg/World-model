# Results component: statistical units versus selected-policy reliability

This is an exposed-development component,not a submission-ready manuscript or
an independent main result. Task:obs8/pred12/rawstride12 on European webcam
detector-silver image-local trajectories. It is not supplemental raw-frame t+50.

We replace only the selected-query loss's all-action-derived scale with the
source-balanced RMS of the quantity actually supervised. Thirty-six matched
risk critics retain network size,initialization,source-balanced draws,step
budget,pointwise loss,weights,forecasts and decoder. Three critic seeds share
the same forecaster seed. The600-update recovery/control pilot and both full
first-head2,000-step refits reproduce exactly. Complete inference/allocation
and a separate full replay agree in all72 groups and144 old-critic controls.

|Query units/context|Known ADE gain over floor|Supported recording-risk views|Known easy violations|
|---|---:|---:|---:|
|Original/local|7.185249%|2/444|401|
|Corrected/local|8.563025%|2/444|396|
|Original/selected-set|6.574842%|0/444|393|
|Corrected/selected-set|8.215347%|1/444|383|

The selected-set known gain rises by1.640505 percentage points
(conditional95% locality bootstrap interval[1.101933,2.251535]),positive in
all12 development localities. However,the prespecified paired wholly-unknown
completion-lower change is-2.597889pp[-7.642133,0.293356],and the complete
recording-risk requirement fails. Unknown selected row views increase despite
fewer total interventions. Correcting loss units improves one utility endpoint
but does not establish robust improvement or safe deployment.

The training diagnosis distinguishes global loss from selection-conditioned
behavior. Original-unit fixed-query monitor error improves,but all-action risk
MSE worsens. A separately registered,exactly replayed score audit confirms
improvement on the identical original selected sets,while final selected sets
retain substantial easy-harm underprediction. Final selected populations differ;
their error contrast is descriptive,not a unique causal explanation. This
motivates a separate fixed-versus-current-selection training experiment,not a
post-hoc risk-threshold revision or an assertion of calibrated decision learning.

Intervals use3,000 locality resamples after repeated context/head-seed averaging.
The74 recordings/149,510 target rows yield reused views,not444 independent
recordings or897,060 independent examples. These nominal intervals do not adjust
for the broader adaptive research programme. Unknown/empty support remains
reported;FDE missing endpoints are not imputed. No learned trajectory-dynamics,
new physical-bank superiority,independent calibration,metric/time,world-model
success or CCF-A-readiness claim follows. Stage5C and SMC remain disabled.
