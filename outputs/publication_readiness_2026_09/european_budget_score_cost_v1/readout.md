# Original-Budget Score Learning: Complete Matched Result

## Material Passport

Fresh_run:162real Torch cost heads,324,000primary updates,2,000extra
full-refit updates,420real pilot updates,216fixed readouts and separate
complete exact policy inference replay. Cached_verified:previous SOURCE
inputs,original controls and frozen old/new active-producer banks.
Not_run:controller-held/independent outcomes,new trajectory producer,
scene-joint decision training or deployment.

Training PID78301:777.652540s,peak RSS10,908,450,816B.
Initial readout PID80961:461.111830s; separate replay PID82735:452.165372s.
All54original full scientific states and all72original prediction/action/
full result controls are exact. The first upper_score new-bank model
has an exact independent2,000step refit. Native arm64 CPU4/workers0,
no NumPy training fallback and no shortened model or update roster.

## Controlled Learning Change

Keep identified five-moment regression and within-row ranking. Add
signed H-0.02R and H_easy-0.02R_easy residual supervision,once with
symmetric squared error and once with4xunderprediction weight.
The original risk rule,decoder,banks,sampler,features and normalizers
are unchanged. No supervised score,target or future mask enters
inference. These are estimated scores,not a calibrated risk theorem.

Unique queries:149,510; recordings:74; localities:12.
Known:146,347; wholly unknown:3,163.
Repeated row views per objective:897,060; all objectives:2,691,180.
Counts across contexts,seeds,objectives and banks are not independent
samples. Intervals use3,000locality resamples after context/head-seed
averaging. SOURCE has historical development exposure; this is fitting
resubstitution,not independent cross-scene confirmation.

## Complete-Policy Accuracy And Recording Risk

| Objective | Bank | Known ADE gain %, CI95 | Required recording risk support | Known easy violations | Undefined easy views | Switch row views |
|---|---|---|---:|---:|---:|---:|
| original | physical_only | +1.946059 [+0.714940, +3.800027] | 42/444 | 313 | 35 | 268083 |
| original | old_neural | +1.968657 [+0.727097, +3.864063] | 42/444 | 307 | 33 | 259875 |
| original | new_neural | +1.893446 [+0.692104, +3.721837] | 37/444 | 313 | 32 | 267092 |
| symmetric_score | physical_only | +3.041956 [+1.240623, +5.574795] | 21/444 | 336 | 33 | 354965 |
| symmetric_score | old_neural | +3.117461 [+1.352072, +5.577540] | 11/444 | 350 | 27 | 354640 |
| symmetric_score | new_neural | +3.015832 [+1.252002, +5.490851] | 19/444 | 350 | 31 | 350086 |
| upper_score | physical_only | +1.752622 [+0.602597, +3.448438] | 50/444 | 288 | 43 | 274190 |
| upper_score | old_neural | +1.851186 [+0.644907, +3.661351] | 37/444 | 304 | 40 | 279005 |
| upper_score | new_neural | +1.979966 [+0.655820, +3.950357] | 41/444 | 312 | 42 | 266028 |

Safety keeps the complete unchanged all/easy2%rule on every required
recording,including missing-outcome bounds. Pooled easy net gain is not
the selected-easy positive-harm risk. Full-roster FDE remains undefined
when a required locality lacks requested final labels. The
[full-reference support census](reference_support.json) distinguishes
available easy-reference mass from the support actually selected by a
policy;neither should be mistaken for a passed safety requirement.

## Paired Objective Effects On The Same Bank

| Objective | Bank | Full known pp, CI95 | Completion lower pp, CI95 | Same-count lower pp, CI95 |
|---|---|---|---|---|
| symmetric_score | physical_only | +1.095897 [+0.487063, +1.802980] | -0.660885 [-3.576588, +1.189885] | not_estimable |
| symmetric_score | old_neural | +1.148804 [+0.577380, +1.808086] | +0.242798 [-0.889981, +1.238101] | not_estimable |
| symmetric_score | new_neural | +1.122386 [+0.494471, +1.865738] | -1.580720 [-6.298195, +1.227017] | not_estimable |
| upper_score | physical_only | -0.193437 [-0.394030, -0.037689] | -0.446285 [-0.696550, -0.230431] | not_estimable |
| upper_score | old_neural | -0.117472 [-0.230655, -0.024787] | -0.499698 [-0.792500, -0.271795] | not_estimable |
| upper_score | new_neural | +0.086520 [-0.164383, +0.413791] | -0.780628 [-2.000011, +0.046352] | not_estimable |

Primary remains upper_score/new_neural minus original/new_neural,
not the best observed arm. Same-count controls measure utility only;
empty comparisons are not_estimable,not safe zero. All contrasts use
actual selected trajectories and the original known floor-reference mass.

- Upper new versus original old,known pp: +0.011309 [-0.172229, +0.264527]
- Upper new versus original old,completion lower pp: -0.493459 [-0.991813, -0.062353]
- Upper new versus symmetric new,known pp: -1.035866 [-1.558396, -0.560069]
- Same upper new head,actual neural-use known pp: +0.209989 [+0.036449, +0.435968]
- Same upper new head,actual neural-use completion lower pp: +0.184456 [+0.013755, +0.405677]

The same-head neural restriction is not another fit. It separates
actual neural path use from shared-head physical-choice changes.

## Actual Training Objective

| Objective | Bank | Heads | First median minibatch loss | Final median minibatch loss |
|---|---|---:|---:|---:|
| original | physical_only | 18 | 2.08923 | 0.333939 |
| original | old_neural | 18 | 1.8956 | 0.324785 |
| original | new_neural | 18 | 1.90402 | 0.332831 |
| symmetric_score | physical_only | 18 | 2.99261 | 0.494977 |
| symmetric_score | old_neural | 18 | 2.69025 | 0.454094 |
| symmetric_score | new_neural | 18 | 2.69531 | 0.449477 |
| upper_score | physical_only | 18 | 5.51454 | 0.532849 |
| upper_score | old_neural | 18 | 4.77388 | 0.562022 |
| upper_score | new_neural | 18 | 4.80534 | 0.504093 |

Objective definitions differ,so loss magnitude is not a comparable
accuracy ranking across arms. Curves are real fitting minibatches,not
validation losses. Unknown rows receive no observed-cost supervision.

## Fixed Fitting-Progress Gates

- complete162fits216groups: true
- exact54original_states72readouts: true
- complete_primary_recording_risk: false
- primary_known_CI_positive: false
- primary_completion_lower_CI_positive: false
- versus_original_old_neural_completion_lower_CI_positive: false
- actual_neural_use_completion_lower_CI_positive: true
- fitting_advancement_gate: false
- independent_confirmation: false
- deployment_changed: false
- submission_ready: false

SOURCE remains supplied detector-silver bbox-midpoint pixels,obs8/pred12/
stride12raw frames. Not historical SDDt50,human gold,verified live
sensor-as-of,metric/seconds,true3D or foundation evidence. No Stage5C
execution orSMC. Independent evidence and scene-joint decision gaps remain.

[Actual curves](figures/summary.svg),[neural-use contrast](figures/neural_use.svg),
[machine evidence](evidence.json),
[failure analysis](failure_analysis.md),[model/data card](model_data_card.md),
[paper addendum](paper_addendum.md),[Chinese guide](reproduction_zh.md).
