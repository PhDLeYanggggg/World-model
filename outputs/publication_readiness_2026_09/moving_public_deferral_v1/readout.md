# Moving Public Regression-Deferral Control

**fresh_run:**54 real Torch routing heads,108000primary updates, complete nested-C evaluation and separate replay. **cached_verified:** fixed forecasters, parent point heads, own-part preprocessing, original splits and data admission. **not_run:** new independent confirmation, new trajectory fitting and deployment.

Required nonempty matched utility is not estimable, not passed. Original full-policy risk/support requirements fail.

## Public Method And What Changed

[Mao, Mohri and Zhong (ICML2024)](https://proceedings.mlr.press/v235/mao24d.html), Section4 Equation3, supplies the two-stage logistic cost-sensitive objective. This is a task adaptation on our fixed seven-action bank, not their original experiment or a new M3W invention. The model directly ranks expected decision costs through a surrogate; it does not fit a one-hot oracle label or report softmax as calibrated confidence. Costs use available-label ADE divided by frozen causal scale. Unknown outcomes are excluded, not labeled zero.

The original point learner used conservative envelope targets for wholly unknown futures. The new public learner excludes those outcome labels. The causal row roster and sampler match, but the effective supervised loss support differs.

All predictors, source roles, 380 active local features, query context, training budget, preprocessing and sampling are unchanged. The public logit output has27969parameters versus28229in the point head. Different head/objective/decision semantics mean this is not an isolated loss-only causal comparison. Unbounded costs, partial labels and causal legal action masks do not automatically inherit the paper theorem or a safety guarantee.

## Direct Development Contrasts

|Public minus point control|Known effect pp [95% CI]|Direct lower pp [95% CI]|
|---|---:|---:|
|public_minus_point|+5.2488 [+1.3068,+9.5513]|-2.4101 [-6.1395,+1.2490]|
|public_veto_minus_point_veto|+0.8168 [-1.0475,+2.9312]|-0.7796 [-2.6794,+0.5941]|

Positive means the public policy has lower available-label error than the point policy. The guarded comparison uses the exact same source-excluded frozen point critic. Full policies can have different coverage, so this is not a matched-ranking test.

## Physical Floor And Absolute Errors

|Policy|Known ADE gain vs floor % [95% CI]|Completion lower % [95% CI]|Known ADE pixels [95% CI]|Endpoint FDE diagnostic pixels [95% CI]|
|---|---:|---:|---:|---:|
|physical_floor|+0.0000 [+0.0000,+0.0000]|+0.0000 [+0.0000,+0.0000]|+61.9408 [+39.9904,+87.2244]|+159.9933 [+101.3015,+231.7458]|
|point_proposer|+7.0459 [+3.1054,+11.7943]|+3.1612 [-0.5821,+7.1900]|+59.3480 [+37.7104,+84.4069]|+151.2953 [+94.3711,+220.8112]|
|point_veto|+3.3298 [+1.0732,+6.9893]|+2.9918 [+0.8972,+6.3508]|+60.7145 [+38.7798,+85.8885]|+156.0955 [+97.8576,+227.2841]|
|point_matched|+3.8926 [+1.3749,+7.7367]|+2.9060 [+0.8373,+5.9009]|+60.5747 [+38.6164,+85.8001]|+155.9244 [+97.5572,+227.0255]|
|public_router|+12.2948 [+6.9366,+17.8779]|+4.2086 [+0.1654,+8.2898]|+56.2059 [+35.3175,+80.3783]|+143.6309 [+89.2998,+209.6765]|
|public_veto|+4.1466 [+1.7752,+7.0181]|+2.5789 [+0.7547,+4.7755]|+60.4919 [+38.6089,+85.7691]|+155.3222 [+97.0074,+226.5137]|
|public_matched|+4.9535 [+2.3490,+8.0205]|+2.1803 [+0.1892,+4.3559]|+60.2551 [+38.2296,+85.6360]|+155.1632 [+96.6152,+226.4049]|

These are secondary source-macro summaries of all original saved policy metrics, not new policies or selected thresholds. A positive contrast against the point learner does not by itself imply positive gain against the physical floor. Available-label ADE and available-endpoint FDE do not impute missing labels. Pixel errors are native image-coordinate diagnostics, not comparable metric distances across cameras. Gain percentages average within-source relative gains; they are not the ratio of the two separately averaged pixel-ADE entries. Easy/hard and tail summaries, including every required NA, are in policy_metrics.json.

## Same-Count Controls

|Veto vs own count/amplitude-matched control|Nonempty known pp [95% CI]|Missing /276|Full-strategy known pp, secondary|Full-strategy lower pp, secondary|
|---|---:|---:|---:|---:|
|point|not estimable|18|-0.5628 [-1.0947,-0.1934]|-1.2739 [-2.0893,-0.5858]|
|public|not estimable|8|-0.8069 [-1.7470,-0.2450]|-2.1758 [-3.6278,-1.1837]|

Required empty matched views remain NA. The original24group primary is still not estimable and not passed. Full-strategy secondary results do not replace it. Intervals use3000 paired source resamples after averaging contexts, seeds and orientations within 23exposed source units. The276source views and696recording views are repeated, not independent.

## Original Full-Policy Risk

|Policy|Supported /696|Known all-risk violations|Known easy-risk violations|Undefined selected easy risk|Switch % [95% CI]|
|---|---:|---:|---:|---:|---:|
|physical_floor|0|0|0|696|+0.0000 [+0.0000,+0.0000]|
|point_proposer|36|494|532|100|+56.6891 [+46.3596,+65.9227]|
|point_veto|44|443|448|173|+26.9609 [+17.2581,+37.2893]|
|point_matched|42|436|445|181|+26.9609 [+17.2581,+37.2893]|
|public_router|14|514|582|50|+65.8277 [+60.5869,+71.5573]|
|public_veto|38|454|483|144|+29.3465 [+21.2958,+37.3791]|
|public_matched|24|440|479|159|+29.3465 [+21.2958,+37.3791]|

The2%positive-harm and easy-net requirements and reference-support conditions are unchanged. No identity certificate is introduced, even for the physical floor. Counts overlap and are not summed. Undefined risks are not zero. Every policy retains all required views.

## Runtime And Evidence

- pilot: PID46124, 42.92s, peak RSS7.156GB.
- train: PID46216, 3306.79s, peak RSS9.800GB.
- evaluate: PID51322, 249.63s, peak RSS5.609GB.
- verify: PID51824, 251.09s, peak RSS5.548GB.

Three real pilot resumed/direct fits and the first full2000-update refit agree exactly. All54fit normalizers, row IDs, sampler states and draw hashes equal the corresponding frozen point controls. All new heads froze before readout. All108policy blocks and prediction hashes match a separate replay. The original point control also reproduces all108original prediction/action blocks and828source-policy readouts exactly. Native arm64 CPU4/interop1/workers0, no resource probing. Optimizer and query sampler states are checkpointed every100updates.

Run `.venv-pytorch/bin/python scripts/run_m3w_moving_public_deferral.py PHASE` for register/pilot/train/evaluate/verify; interrupted unchanged phases accept `--resume`. Do not overwrite completed phases. Local weights and arrays are not committed.

Observation8/prediction12 annotation order in image pixels. This is not verified seconds, metric, raw-frame t50, physical-safety, true3D or foundation evidence. Stage5C and SMC remain off.
