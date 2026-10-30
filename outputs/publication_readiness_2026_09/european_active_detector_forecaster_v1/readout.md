# Detector History On The Actual Active Trajectory Backbone

## Material Passport

Fresh_run: 36 real Torch trajectory producers, 144,000 primary updates,
4,000 full-refit updates, 900 pilot updates, 36 fixed inference groups and a
separate complete exact inference replay. Cached_verified: all 9 original
active grouped/partial-neighbor dimensionless producers, 318,969 SOURCE
rows and the verified 163-record past-only detector store. Not_run: a new
learned gain/harm controller on this proposal bank, controller-held outcomes,
independent selection/calibration/confirmation, DroneCrowd or deployment.

This is the backbone actually consumed by the cost learner. The earlier
72-model flat complete-neighbor experiment remains a separate result, not
a matched replacement of that active neural candidate. All 9 actual parent
designs are retained; there is no active complement-parent family to drop.

Training PID61609: 1825.845342s; peak RSS2,203,729,920B.
Readout PID63955: 556.340011s; separate replay PID64786: 551.535353s.
All 9 complete disabled scientific states and fresh predictions reproduce
their actual parents. First full history refit is exact. Native arm64 CPU 4,
workers 0, no resource probing, no NumPy training fallback or budget cut.

## Fixed Matched Trajectory Accuracy

Gain is relative to each producer own-fit-selected causal reference.
Primary is history, not a winning seed or a post-readout ensemble. Three
producer seeds and 3,000 locality resamples follow fold/seed averaging over
all 12 required localities. All SOURCE readouts are development-exposed;
locality exclusion and exact replay do not restore independent testing.

| Detector input | Known ADE gain %, CI95 | FDE gain %, CI95 | Hard ADE gain %, CI95 | Easy ADE gain %, CI95 |
|---|---|---|---|---|
| disabled | +9.325561 [+6.387427, +12.714358] | not_estimable | +8.547697 [+5.307813, +12.004914] | -9.003002 [-36.189404, +7.697650] |
| current | +8.434376 [+5.728776, +11.938455] | not_estimable | +8.731136 [+6.850901, +10.798855] | -8.955560 [-31.111040, +4.496324] |
| noise | +9.003217 [+5.907696, +12.497459] | not_estimable | +8.266484 [+5.043396, +11.729235] | -9.621559 [-36.875384, +6.884588] |
| history | +10.485965 [+7.859327, +13.812229] | not_estimable | +10.094833 [+6.106546, +13.868006] | -12.222201 [-40.664363, +5.842319] |

## Support And Missing Outcomes

Unique queries: 149,510; recordings: 74; repeated producer/seed row views: 897,060.
Unique complete: 91,609; partial: 54,738; wholly unknown: 3,163.
Counts are not multiplied by input arms. Repeated windows and producer
views are not independent samples. Available-label ADE on a partial row
does not reveal its full 12-step path. Locality 082 has no requested final
labels; undefined full-roster FDE/complete-future views are retained, not
dropped or replaced by zero. A separate exact raw-label audit found no
importer omission and no recoverable exact raw labels on these IDs.

## History Minus Input Controls

| Control | Known ADE change pp, CI95 | Completion-lower pp, CI95 | Completion-upper pp, CI95 |
|---|---|---|---|
| disabled | +1.160404 [-0.355089, +2.540277] | -18.258091 [-45.071627, -2.865096] | +20.578900 [+6.973659, +45.089506] |
| current | +2.051589 [+0.863443, +3.128546] | -14.591686 [-37.448406, -1.795863] | +18.694864 [+6.441143, +40.012207] |
| noise | +1.482748 [-0.149006, +2.933599] | -18.041296 [-45.078105, -2.502519] | +21.006792 [+7.342113, +45.682487] |

Completion envelopes use causal forecast disagreement for wholly
unknown outcomes on the original known reference-mass denominator.
They are bounds, not observed losses or identified full-population gains.
Hard/easy evaluation membership is observed-label diagnostic membership,
not identified missing membership. No future mask or endpoint is an input.

## All Six Preregistered Causal Physical Baselines

| Fixed baseline | Known ADE gain %, CI95 | FDE gain %, CI95 | History-minus-fixed known ADE pp, CI95 | Completion-lower pp, CI95 |
|---|---|---|---|---|
| constant_position | -117.340885 [-163.448859, -72.934336] | not_estimable | +127.826850 [+82.226432, +175.158691] | +92.159819 [+59.547884, +127.457904] |
| constant_velocity_causal_fd | +2.751314 [-4.525851, +11.032768] | not_estimable | +7.734651 [+0.826611, +13.471995] | -16.275312 [-46.355130, +2.390172] |
| damped_velocity_090 | +1.115445 [-1.282809, +3.112073] | not_estimable | +9.370520 [+5.473079, +14.023662] | -1.500071 [-8.420703, +4.891517] |
| damped_velocity_097 | +6.436269 [+0.816192, +12.629448] | not_estimable | +4.049696 [-1.214258, +8.322004] | -13.386860 [-34.170438, -0.409080] |
| history_ols4_velocity | +1.279565 [-4.335622, +8.835853] | not_estimable | +9.206400 [+4.148105, +13.357641] | -16.745690 [-46.602884, +1.931364] |
| history_ols8_velocity | -20.656348 [-35.445527, -7.245246] | not_estimable | +31.142312 [+19.527710, +44.934949] | +9.029062 [-1.667782, +18.827961] |

Physical contrasts were specified before scores. Differences use the
original own-fit reference mass, not a new percent denominator for each
physical method. No reference, model, threshold or alpha is chosen here.

## Complete Locality Breakdown

| Locality | Unique queries | Last-label rows | History-minus-disabled known ADE pp | Raw history easy gain % |
|---|---:|---:|---:|---:|
| eu-locality-007 | 1089 | 515 | -3.314199 | +0.100850 |
| eu-locality-008 | 81958 | 61617 | +2.539370 | -0.416118 |
| eu-locality-020 | 415 | 228 | +2.121027 | +29.776928 |
| eu-locality-048 | 1470 | 983 | +0.559455 | +5.688995 |
| eu-locality-067 | 3631 | 2525 | +1.122016 | -2.232252 |
| eu-locality-074 | 55773 | 43522 | +2.459144 | +0.533875 |
| eu-locality-082 | 78 | 0 | -4.176434 | -157.308372 |
| eu-locality-110 | 1060 | 878 | +5.398304 | +0.761521 |
| eu-locality-112 | 506 | 316 | +1.943289 | +0.951714 |
| eu-locality-119 | 1658 | 1193 | +3.062252 | -1.482673 |
| eu-locality-124 | 281 | 148 | -0.384993 | -21.422930 |
| eu-locality-126 | 1591 | 1240 | +2.595622 | -1.617947 |

All required localities remain, including small or partially observed
ones. These are development diagnostics, not per-locality safety proofs.

## Actual Training Loss

| Input | Models | First median minibatch objective | Final median minibatch objective |
|---|---:|---:|---:|
| disabled | 9 | 1.06572 | 0.966489 |
| current | 9 | 1.06572 | 0.94682 |
| noise | 9 | 1.06572 | 0.966346 |
| history | 9 | 1.06572 | 0.940728 |

These are actual training minibatches, not validation or test losses.
The legacy trace field mean_past_normalized_ADE contains restored pixel
ADE before own-fit loss factors in this SOURCE adapter, not metres.
The objective is differentiated through the active temporal/interaction
Transformer; 512 new detector weights do not replace its original topology.

## Proposal Evidence Gates

- all36groups: true
- all9fresh_parent_predictions_exact: true
- history_minus_disabled_known_CI_positive: false
- history_minus_current_known_CI_positive: true
- history_minus_noise_known_CI_positive: false
- history_minus_disabled_completion_lower_CI_positive: false
- history_minus_current_completion_lower_CI_positive: false
- history_minus_noise_completion_lower_CI_positive: false
- proposal_input_evidence_gate: false
- learned_policy_safety_evaluated: false
- deployment_changed: false
- submission_ready: false

No raw-proposal gate certifies the original 2% complete all/easy policy-risk
requirement on every required recording. Empty selected risks stay not_estimable.
A learned intervention policy must be trained and evaluated separately.

## Scope And Limits

SOURCE supplied detector-silver bbox midpoints, obs8/pred12/stride12 raw
frames, pixel coordinates. Not human gold, verified live sensor-as-of, metres,
seconds, true 3D or a foundation model. The branch combines shape, confidence,
body-relative displacement and overlap; it does not isolate a confidence-
only mechanism. No new scene image, JEPA, generative rollout, Stage5C or SMC.
No independent roles or outcomes from the other 89 recordings open. Local
exact replay does not complete an anonymous from-scratch asset release.

[Aggregate figure](figures/summary.svg), [actual loss traces](training_traces.json),
[evidence](evidence.json), [model/data card](model_data_card.md),
[failure analysis](failure_analysis.md), [Chinese reproduction guide](reproduction_zh.md).
