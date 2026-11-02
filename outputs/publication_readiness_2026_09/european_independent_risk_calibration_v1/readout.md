# Independent Risk Calibration

Result source: fresh_run dedicated-role conversion, frozen Torch inference,
allocation and scoring; cached_verified SOURCE-trained weights and source-only
normalization. No new optimizer updates. The selection scenes are not reused
as calibration scenes. Final confirmation and DroneCrowd remain closed.

## Complete Cohort And Decision

102 recordings in 12 locality groups; nine frozen pipelines, hence 918
repeated pipeline-recording views, not 918 independent recordings.
All 13 multipliers and four allocation policies freeze before labels.
Original all/easy positive-harm and easy-net budgets remain 2%.

Selected common multiplier: **none**.
All required calibration views supported: **False**.
If no multiplier qualifies, the global physical floor is the separate fallback,
not a calibrated learned success. No deployment promotion is made here.

## Fixed-Grid Readout

|Lambda|Policy|Known ADE gain % [conditional locality CI]|Completion-lower gain %|Supported / 918|Undefined views|Worst easy net %|
|---:|---|---:|---:|---:|---:|---:|
|1|frame_uniform|1.025337 [0.706990, 1.361252]|0.600035|27/918|170|50.662242|
|1|independent|1.412052 [1.092173, 1.775720]|0.839320|6/918|9|41.079537|
|1|joint_free|2.257994 [1.719062, 2.862776]|1.196263|1/918|9|110.673560|
|1|joint_matched|1.925481 [1.439274, 2.485502]|1.004763|0/918|9|103.571491|
|2|frame_uniform|0.642271 [0.414795, 0.906837]|0.382411|33/918|242|26.830007|
|2|independent|0.836721 [0.612274, 1.086833]|0.479615|22/918|9|20.934704|
|2|joint_free|1.477507 [1.066187, 1.925174]|0.833851|4/918|9|73.863446|
|2|joint_matched|1.245360 [0.872599, 1.656413]|0.681344|6/918|9|68.285418|
|4|frame_uniform|0.374189 [0.222875, 0.549080]|0.227647|51/918|327|10.418165|
|4|independent|0.469996 [0.318701, 0.634924]|0.238713|38/918|10|10.968315|
|4|joint_free|0.894404 [0.630587, 1.155254]|0.503349|13/918|10|47.533719|
|4|joint_matched|0.742021 [0.515971, 0.968074]|0.413664|21/918|10|45.246503|
|8|frame_uniform|0.211287 [0.112241, 0.328202]|0.124680|85/918|407|9.511221|
|8|independent|0.270900 [0.162754, 0.384549]|0.116636|48/918|15|9.949811|
|8|joint_free|0.502022 [0.338318, 0.665571]|0.253630|23/918|14|28.895148|
|8|joint_matched|0.393723 [0.272137, 0.515292]|0.191822|30/918|16|28.043613|
|16|frame_uniform|0.121075 [0.053438, 0.205711]|0.076920|95/918|503|5.448204|
|16|independent|0.159951 [0.090168, 0.236089]|0.063277|74/918|23|11.588328|
|16|joint_free|0.256360 [0.157228, 0.367793]|0.104002|50/918|21|17.173477|
|16|joint_matched|0.211727 [0.131425, 0.300101]|0.085266|64/918|23|16.809142|
|32|frame_uniform|0.061454 [0.021939, 0.112629]|0.038273|96/918|596|3.320551|
|32|independent|0.101288 [0.047712, 0.160729]|0.049349|113/918|43|7.926977|
|32|joint_free|0.141698 [0.079631, 0.210035]|0.057313|93/918|40|10.467753|
|32|joint_matched|0.115752 [0.061074, 0.176699]|0.055605|99/918|43|10.251702|
|64|frame_uniform|0.036020 [0.008547, 0.072479]|0.026810|75/918|694|2.585073|
|64|independent|0.051781 [0.026324, 0.083089]|0.020822|150/918|80|8.509254|
|64|joint_free|0.087237 [0.042545, 0.136473]|0.047775|124/918|77|8.890900|
|64|joint_matched|0.070577 [0.034981, 0.109666]|0.034137|126/918|80|8.918293|
|128|frame_uniform|0.020608 [0.002536, 0.044512]|0.018085|62/918|777|1.625995|
|128|independent|0.027846 [0.012789, 0.049204]|0.008902|216/918|146|1.950797|
|128|joint_free|0.047505 [0.023099, 0.076005]|0.024891|203/918|138|8.509254|
|128|joint_matched|0.039256 [0.018388, 0.063832]|0.019935|192/918|146|8.457342|
|256|frame_uniform|0.011652 [0.000478, 0.026412]|0.009130|51/918|821|0.706588|
|256|independent|0.015081 [0.004685, 0.030278]|-0.000309|255/918|221|1.950797|
|256|joint_free|0.025154 [0.010218, 0.043306]|0.009991|225/918|214|1.950797|
|256|joint_matched|0.018570 [0.007249, 0.033534]|0.003348|229/918|222|1.950797|
|512|frame_uniform|0.002749 [-0.000129, 0.007679]|0.002735|35/918|865|0.095346|
|512|independent|0.008773 [0.002455, 0.016185]|0.001405|279/918|316|0.329393|
|512|joint_free|0.009565 [0.002429, 0.018299]|0.002298|262/918|309|1.098538|
|512|joint_matched|0.008824 [0.002450, 0.016601]|0.001587|265/918|315|0.896094|
|1024|frame_uniform|0.001471 [-0.000096, 0.004507]|0.001470|27/918|885|0.033654|
|1024|independent|0.004329 [0.000940, 0.008942]|0.001676|266/918|426|0.315953|
|1024|joint_free|0.006510 [0.002118, 0.011632]|0.003824|258/918|423|0.543891|
|1024|joint_matched|0.004458 [0.001263, 0.008497]|0.001803|254/918|426|0.452608|
|2048|frame_uniform|0.001519 [0.000000, 0.004551]|0.001519|19/918|898|0.023386|
|2048|independent|0.002528 [0.000315, 0.005855]|-0.000181|206/918|567|0.090244|
|2048|joint_free|0.003728 [0.000819, 0.007327]|0.001018|203/918|558|0.201620|
|2048|joint_matched|0.002532 [0.000321, 0.005838]|-0.000179|201/918|567|0.178538|
|4096|frame_uniform|0.001515 [-0.000000, 0.004546]|0.001515|15/918|903|0.008462|
|4096|independent|0.001875 [0.000135, 0.004708]|-0.000513|142/918|677|0.119078|
|4096|joint_free|0.002105 [0.000123, 0.005389]|-0.000284|145/918|674|0.202904|
|4096|joint_matched|0.002099 [0.000121, 0.005390]|-0.000290|141/918|675|0.201160|

## Interpretation

These are in-sample calibration outcomes after fitting the common multiplier.
The locality bootstrap describes conditional uncertainty; it does not turn
calibration fitting into independent efficacy or a population risk certificate.
Twelve localities, correlated recordings, ratio denominators, partial labels
and reselected/nonmonotone actions do not justify applying a general conformal
risk guarantee without its assumptions and an appropriate loss/test procedure.
See [Learn then Test](https://arxiv.org/abs/2110.01052) for hypothesis-testing
risk control and [Conformal Risk Control](https://arxiv.org/abs/2208.02814)
for monotone-loss expected-risk calibration; neither theorem is claimed here.

Available-label ADE is not full12 ADE. Completion bounds cover wholly unknown
rows only, not missing steps inside partial trajectories. Endpoint FDE is
diagnostic if missing. Released tracker/detector silver is not human gold.
Inputs are retrospective past-only, not a proof of online sensor causality.
Local image pixels and raw stride12 are not metric units or seconds.
This is not true3D, a foundation model, Stage5C execution or SMC.

## Execution Receipts

|Phase|PID|Seconds|Peak RSS bytes|
|---|---:|---:|---:|
|pilot|4339|49.672492|2365243392|
|build|4554|204.410813|2762833920|
|verify_inputs|5028|207.244544|3306864640|
|infer|5393|827.282159|2431598592|
|verify_inference|6966|837.194931|2397880320|
|policy|8905|12106.168735|853803008|
|labels|23238|162.685442|3451518976|
|eval|23672|989.094563|3310239744|
|verify|25146|1066.039289|3259449344|

Registration, phase receipts and scalar curves are public. All weights,
input/feature arrays, labels and policy matrices remain private.
Fresh raw/input replay, fresh model/cost replay, metric/feasibility replay
and MILP reoptimization are distinct checks. See verify.json for exact scope.
Original negative SOURCE/model-selection experiments remain unchanged.
