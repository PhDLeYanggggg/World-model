# Past Detector Histories: Matched Cost Learning

## Material Passport

Fresh_run:144 real Torch cost heads,288,000 primary optimizer updates,
2,000 extra full-refit updates,360 pilot updates,288 fitting groups and a
separate complete inference replay. Cached_verified: frozen trajectory
producer seed43, original fitting packets, normalizers and raw-replayed
detector features. Not_run: new trajectory/world-dynamics training, held
or independent evaluation, deployment. No NumPy training fallback.

Training PID41389, exit0: 605.155608s, peak RSS10,198,335,488B.
Readout PID43421: 277.014973s; replay PID44412: 268.294171s.
All36disabled endpoints exactly reproduce old known-only model/Adam/RNG/
draws/traces. The first full history/mixed refit is exact. Four representations
share the original398fields and64-field zero-initial branch;30,405parameters.
Extra input statistics use own-context causal fitting rows, never future support.

## Complete Policy And Risk

| Representation | Bank | Known ADE gain %, fitting CI95 | Risk support | Undefined easy | Known all/easy violations | Selected / unknown |
|---|---|---|---:|---:|---:|---:|
| disabled | physical_only | +0.578053 [+0.227296, +1.033762] | 90/444 | 55 | 75/206 | 73184 / 851 |
| disabled | physical_neural | +0.580799 [+0.270073, +0.978949] | 77/444 | 62 | 85/234 | 80414 / 932 |
| current | physical_only | +1.036997 [+0.420433, +1.889014] | 35/444 | 52 | 145/291 | 144756 / 2058 |
| current | physical_neural | +1.103071 [+0.444003, +1.990557] | 25/444 | 45 | 172/313 | 148868 / 2077 |
| noise | physical_only | +0.671289 [+0.226560, +1.343743] | 127/444 | 67 | 35/169 | 59527 / 674 |
| noise | physical_neural | +0.637508 [+0.261332, +1.177228] | 108/444 | 70 | 50/210 | 68362 / 814 |
| history | physical_only | +1.946059 [+0.714940, +3.800027] | 42/444 | 35 | 161/313 | 268083 / 3532 |
| history | physical_neural | +1.968657 [+0.727097, +3.864063] | 42/444 | 33 | 163/307 | 259875 / 3445 |

Known-label gains are not complete-outcome safety. All444recording views
per representation repeat six contexts/head seeds over74unique recordings.
Undefined selected/easy risk is not a passed zero. No recording is removed.

## Primary History Minus Information Controls

| Control | Full known change pp, CI95 | Full completion-lower pp, CI95 | Matched completion-lower pp, CI95 |
|---|---|---|---|
| disabled | +1.387858 [+0.413289, +2.887799] | +0.972969 [-0.108698, +2.535644] | +0.610170 [-0.064770, +1.640963] |
| current | +0.865586 [+0.253824, +1.878385] | +0.416683 [-0.324895, +1.535674] | +0.404823 [-0.155446, +1.308032] |
| noise | +1.331149 [+0.370384, +2.721853] | +0.896374 [-0.169930, +2.390134] | +0.529823 [-0.048068, +1.382448] |

Missing-outcome disagreement mass remains in completion bounds, with fixed
known floor reference mass. Count-matched controls compare utility only.
All intervals use3,000locality bootstrap draws after averaging contexts and
controller seeds across12localities. They are fitting-conditional, not
independent confirmation or whole-pipeline OOF. Noise can encode identity;
it controls capacity/keys, not useful visual evidence.

## Neural Candidate Bank Increment

| Representation | Mixed minus physical known pp, CI95 | Completion-lower pp, CI95 |
|---|---|---|
| disabled | +0.002746 [-0.066949, +0.064489] | -0.033985 [-0.095423, +0.013719] |
| current | +0.066074 [-0.008190, +0.161235] | -0.054280 [-0.129328, +0.044254] |
| noise | -0.033781 [-0.188387, +0.078227] | -0.086240 [-0.240487, +0.026850] |
| history | +0.022598 [-0.032990, +0.090980] | -0.301435 [-0.520784, -0.131315] |

## Actual Minibatch Loss

| Representation | Physical first / final median | Mixed first / final median |
|---|---|---|
| disabled | 2.08923 / 0.481839 | 1.8956 / 0.45454 |
| current | 2.08923 / 0.415888 | 1.8956 / 0.402811 |
| noise | 2.08923 / 0.300547 | 1.8956 / 0.301041 |
| history | 2.08923 / 0.333939 | 1.8956 / 0.324785 |

First/final minibatches are not fixed validation losses. Falling objective
does not certify selected-risk calibration. All traces are retained.

## FDE And Locality Sensitivity

| Representation | Mixed known FDE gain %, fitting CI95 |
|---|---|
| disabled | not_estimable |
| current | not_estimable |
| noise | not_estimable |
| history | not_estimable |

Any required zero FDE reference remains not_estimable; no locality is
deleted to manufacture an aggregate. ADE locality values below are known
outcomes, not independent tests or cross-camera pooled pixel errors.

| Locality | Disabled mixed gain % | Current mixed gain % | Noise mixed gain % | History mixed gain % |
|---|---:|---:|---:|---:|
| eu-locality-007 | +0.739506 | +0.932217 | +0.876472 | +1.239778 |
| eu-locality-008 | +0.737835 | +1.347782 | +0.689709 | +2.639994 |
| eu-locality-020 | +0.866024 | +0.912889 | +1.224044 | +1.175826 |
| eu-locality-048 | +0.119262 | +0.138710 | +0.243327 | +0.673363 |
| eu-locality-067 | +0.404059 | +0.507335 | +0.457303 | +1.152601 |
| eu-locality-074 | +0.449119 | +0.803827 | +0.063822 | +0.863183 |
| eu-locality-082 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| eu-locality-110 | +2.408308 | +5.130615 | +3.158270 | +11.072281 |
| eu-locality-112 | +0.079986 | +0.194476 | +0.090774 | +0.429096 |
| eu-locality-119 | +0.153913 | +0.434362 | +0.306865 | +0.639693 |
| eu-locality-124 | +0.042109 | +0.056441 | +0.052512 | +0.088086 |
| eu-locality-126 | +0.969471 | +2.778194 | +0.487004 | +3.649984 |

## Gates

- full288groups: true
- complete_original_recording_risk: false
- history_minus_disabled_full_lower_CI_positive: false
- history_minus_disabled_matched_lower_CI_positive: false
- history_minus_current_full_lower_CI_positive: false
- history_minus_current_matched_lower_CI_positive: false
- history_minus_noise_full_lower_CI_positive: false
- history_minus_noise_matched_lower_CI_positive: false
- mixed_minus_physical_lower_CI_positive: false
- fitting_advancement_gate: false
- independent_confirmation: false
- deployment_changed: false
- submission_ready: false
- stage5c_executed: false
- smc_enabled: false

The primary remains history/mixed; no secondary replaces it after scoring.
SOURCE is oblique detector-silver midpoint pixels,obs8/pred12/stride12 raw
frames. Acquisition-as-of is not certified. No metric,seconds,human-gold,
true3D,foundation,Stage5C or SMC claim. Independent roles and DroneCrowd stay closed.

[Actual loss and policy figures](figures/summary.svg), [machine-readable evidence](evidence.json),
[complete training traces](training_traces.json), [reproduction](reproduction_zh.md).
