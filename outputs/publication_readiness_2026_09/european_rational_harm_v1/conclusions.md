# Rational Bounded Harm: Complete Matched Experiment

## Material Passport

Fresh18 fits,36,000 updates in190.94s;144 readout views in122.32s.
Seven old controls are cached_verified models with fresh exactly matching
inference. No old trajectory predictor or new data role was trained/opened.
This is not independent confirmation.
The18 new finals were all frozen before this readout. No checkpoint or
threshold selection. Native arm64 CPU4/workers0, read-only CREATE TRAIN
streaming; no new scheduler job.

## Controlled Change

The harm branch uses an anchor-exact C1 rational mapping instead of
hard-clipped increments. Benefit/reference outputs remain
exactly fixed. Initial weights, source/query sampling, draw hashes, optimizer
settings, budget and12,258 trainable/25,407 total parameters match the
isolated-odds and hard-clipped controls. Mapping geometry and tail approach
both change; they are not uniquely isolated. This is a cost head, not a
new dynamics forecaster. Boundary/underflow gradients can still be flat.

Utility continuation, joint risk, isolated odds and full-range have4,000
effective head updates including their2,000-step anchors. Local, anchor
and old-risk controls have2,000 updates and are not equal-total-budget
controls. No favorable comparator is omitted.

## Complete Results

### Fit

| Policy | Signed MSE | Selected occurrences | Known easy violations /72 | Upper easy violations /72 | Empty | Undefined easy | Full support /72 |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 0.559739 | 123490 | 1 | 59 | 1 | 7 | 5 |
| anchor | 0.489310 | 337984 | 62 | 65 | 4 | 6 | 1 |
| old_risk | 1.965634 | 67774 | 9 | 57 | 3 | 7 | 7 |
| utility_continue | 0.396041 | 401255 | 65 | 66 | 1 | 6 | 0 |
| joint_risk | 1.224959 | 76621 | 6 | 62 | 3 | 6 | 4 |
| isolated_risk | 1.985264 | 45395 | 21 | 54 | 5 | 6 | 11 |
| full_range | 2.258883 | 58707 | 25 | 66 | 0 | 6 | 0 |
| rational | 2.341136 | 48990 | 20 | 64 | 0 | 8 | 0 |

| Rational minus control | Full lower utility pp [95% CI] | Matched-count lower utility pp [CI] | Signed MSE difference [CI] |
|---|---:|---:|---:|
| local | -3.820815 [-7.024726, -1.392839] | -0.692460 [-1.451585, -0.123332] | 1.781397 [1.003328, 2.734659] |
| anchor | -2.987299 [-5.935096, -0.720600] | 0.258441 [-0.311095, 1.065994] | 1.851827 [1.085946, 2.761760] |
| old_risk | -0.694208 [-4.045997, 1.736203] | 0.368050 [0.028895, 0.765261] | 0.375502 [-0.450280, 1.221414] |
| utility_continue | -5.045371 [-8.622113, -2.415571] | -0.412884 [-0.841423, -0.080359] | 1.945095 [1.179491, 2.855438] |
| joint_risk | -1.219977 [-3.897913, 0.622163] | 0.175971 [-0.002535, 0.396600] | 1.116177 [0.531885, 1.931430] |
| isolated_risk | 0.271053 [-2.896367, 2.681767] | 0.352854 [0.011114, 0.761193] | 0.355872 [0.034032, 0.698364] |
| full_range | -3.633630 [-6.657561, -1.789977] | -0.485777 [-0.759431, -0.239003] | 0.082253 [-0.302171, 0.417235] |

### Held

| Policy | Signed MSE | Selected occurrences | Known easy violations /72 | Upper easy violations /72 | Empty | Undefined easy | Full support /72 |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 4.013844 | 57903 | 46 | 65 | 3 | 7 | 0 |
| anchor | 1.695168 | 154772 | 61 | 71 | 0 | 1 | 0 |
| old_risk | 7.452018 | 27826 | 40 | 56 | 5 | 12 | 1 |
| utility_continue | 1.967093 | 182163 | 61 | 70 | 0 | 2 | 0 |
| joint_risk | 4.663019 | 36085 | 45 | 60 | 4 | 9 | 0 |
| isolated_risk | 5.047722 | 24204 | 47 | 63 | 3 | 4 | 3 |
| full_range | 7.774976 | 31270 | 44 | 59 | 4 | 12 | 0 |
| rational | 6.644551 | 27371 | 34 | 51 | 3 | 19 | 0 |

| Rational minus control | Full lower utility pp [95% CI] | Matched-count lower utility pp [CI] | Signed MSE difference [CI] |
|---|---:|---:|---:|
| local | -2.015354 [-4.115769, -0.265790] | -0.072735 [-0.443363, 0.287799] | 2.630707 [0.411296, 5.107616] |
| anchor | -2.131965 [-3.336372, -1.008992] | 0.147118 [-0.321367, 0.780996] | 4.949383 [2.689308, 7.476471] |
| old_risk | -0.041091 [-1.411638, 1.204976] | 0.151913 [0.032439, 0.286865] | -0.807467 [-3.118304, 1.179369] |
| utility_continue | -3.046028 [-4.270168, -1.879701] | -0.223094 [-0.507980, 0.042641] | 4.677458 [2.579641, 7.023329] |
| joint_risk | -0.572903 [-1.374259, 0.173355] | 0.058333 [-0.101095, 0.226338] | 1.981532 [0.938362, 3.105085] |
| isolated_risk | 0.026904 [-1.427016, 1.408811] | 0.201973 [-0.029673, 0.473714] | 1.596829 [0.469571, 2.899709] |
| full_range | -1.547315 [-2.429844, -0.825328] | -0.217080 [-0.376888, -0.062234] | -1.130425 [-2.701244, 0.202614] |

## Registered Gate

Advance to later development design: **False**.
Failures: local_full_not_positive, local_matched_not_positive, anchor_full_not_positive, anchor_matched_not_positive, old_risk_full_not_positive, utility_continue_full_not_positive, utility_continue_matched_not_positive, joint_risk_full_not_positive, joint_risk_matched_not_positive, isolated_risk_full_not_positive, isolated_risk_matched_not_positive, full_range_full_not_positive, full_range_matched_not_positive, full_policy_support_failed.

Positive lower95% intervals required against every control at both full
and matched counts, plus full-policy risk support in all72 held views.
Empty/undefined or incomplete-label policies do not earn a safety pass.
A lower loss/MSE or one positive source cannot substitute for this rule.

Utility is pair-specific completion-lower mass divided by full-known
reference mass, NOT ADE/FDE. Bounds are not transitive. Original TRAIN
fit/held recordings are already exposed. Bootstrap3,000 averages repeated
contexts/seeds within12 localities; nominal, dependent and not multiplicity
adjusted or independent confirmation. Three cost seeds are not three
new forecasting producers. No deployable world-model success is inferred.

## Reproduction

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_rational_harm check
.venv-pytorch/bin/python -m scripts.report_m3w_rational_harm --check
```

Complete outputs are immutable; only unfinished phases permit --resume.
Private checkpoints and raw histories are excluded from Git. Public outputs
contain aggregate metrics, provenance and reproducible code.

Obs8/pred12 raw stride12, detector-track pixel/local coordinates. No metric,
seconds, physical safety, human gold, true3D or foundation claim.
Stage5C/SMC disabled. Historical Stage37 is exploratory, not recertified.
