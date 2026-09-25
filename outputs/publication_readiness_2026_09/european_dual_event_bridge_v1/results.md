# Dual-Event Policy Bridge Results

## Material Passport
Fresh real neural-cost training and frozen-decision readout; cached_verified source forecasts.
Reused opened model-selection evidence only. Six localities, not 18 independent repetitions.

Primary comparator: the preceding old easy add_only policy, not the weaker bridge reference.

| Policy | All ADE gain | Hard ADE gain | Gain vs training-selected motion | Worst-locality easy degradation | Easy passes | Positive/negative CI |
|---|---:|---:|---:|---:|---:|---:|
| all_risk_only | +3.947701% to +5.566952% | +4.390476% to +6.286272% | +2.893989% to +11.358978% | +0.000000% to +0.730345% | 18/18 | 18/0 |
| candidate_all | +4.768435% to +5.953220% | +5.050799% to +7.439060% | +3.094065% to +11.734089% | +0.000000% to +6.076432% | 12/18 | 18/0 |
| dual_risk | +0.000610% to +1.778578% | -0.042511% to +1.666554% | -1.689162% to +7.101015% | +0.000000% to +0.000000% | 18/18 | 17/0 |
| easy_risk_only | +0.029743% to +1.867776% | -0.029794% to +1.740736% | -1.683729% to +7.168568% | +0.000000% to +0.000000% | 18/18 | 17/0 |
| old_all_add | +4.880257% to +6.475198% | +5.169725% to +8.035468% | +3.232387% to +12.397306% | +0.000000% to +8.216907% | 11/18 | 18/0 |
| old_easy_add | +0.000000% to +0.000000% | +0.000000% to +0.000000% | -1.773321% to +6.636237% | +0.000000% to +0.000000% | 18/18 | 0/0 |
| raw_neural | -9.754866% to +12.887175% | +6.258250% to +18.653663% | -2.807077% to +11.411586% | +24.774235% to +30.675096% | 0/18 | 6/12 |
| reference_easy | -0.295888% to -0.010531% | -0.103573% to +0.004322% | -1.829906% to +6.360054% | +0.000000% to +0.000000% | 18/18 | 0/16 |
| ridge_dual | -0.008534% to +1.872164% | +0.127967% to +1.910135% | -0.884775% to +7.042027% | +0.000000% to +0.000000% | 18/18 | 12/0 |
| training_selected | -7.744306% to +1.471370% | +5.003661% to +9.266495% | +0.000000% to +0.000000% | +13.717222% to +36.908965% | 0/18 | 0/12 |
| utility_only | +4.740680% to +5.977372% | +5.038034% to +7.538681% | +3.086181% to +11.758556% | +0.000000% to +6.076432% | 15/18 | 18/0 |

Ranges cover all registered groups, not selected winners. All/easy/hard labels retain
producer-training thresholds. CV-zero harm is separate. Endpoint and complete-label results,
p95/p99 errors and per-locality costs are in the group JSON files.

## Three-Seed Paired Ablations

Mean seedwise relative gains within each locality, followed by 3,000 locality resamples.
No trajectory ensembling, window-level independence claim or multiplicity adjustment.

| Producer/controller | Dual vs control | All gain | 95% locality CI | Hard gain |
|---|---|---:|---:|---:|
| producer0_controller1 | utility_only | -5.696001% | -7.916271% to -3.647691% | -6.904896% |
| producer0_controller1 | all_risk_only | -5.278351% | -7.222708% to -3.487006% | -6.345585% |
| producer0_controller1 | easy_risk_only | -0.027596% | -0.037228% to -0.018068% | -0.015971% |
| producer0_controller1 | ridge_dual | -0.003628% | -0.287919% to +0.204926% | -0.184707% |
| producer0_controller2 | utility_only | -5.550796% | -7.769653% to -3.628311% | -6.659347% |
| producer0_controller2 | all_risk_only | -5.082306% | -7.113992% to -3.229157% | -5.932270% |
| producer0_controller2 | easy_risk_only | -0.041769% | -0.063437% to -0.018293% | -0.053927% |
| producer0_controller2 | ridge_dual | +0.236122% | +0.144958% to +0.327462% | -0.023259% |
| producer1_controller0 | utility_only | -5.544566% | -7.507790% to -3.656572% | -7.595068% |
| producer1_controller0 | all_risk_only | -4.967920% | -6.946863% to -3.111742% | -6.287388% |
| producer1_controller0 | easy_risk_only | -0.017746% | -0.048283% to +0.007895% | -0.004047% |
| producer1_controller0 | ridge_dual | -0.403615% | -0.837191% to +0.027713% | -0.505664% |
| producer1_controller2 | utility_only | -5.565800% | -7.757597% to -3.658419% | -7.591333% |
| producer1_controller2 | all_risk_only | -4.195637% | -5.920574% to -2.508723% | -5.070637% |
| producer1_controller2 | easy_risk_only | -0.012727% | -0.026670% to -0.000705% | -0.029207% |
| producer1_controller2 | ridge_dual | +0.101898% | -0.113355% to +0.271427% | -0.225865% |
| producer2_controller0 | utility_only | -4.920777% | -6.765991% to -3.111229% | -5.381912% |
| producer2_controller0 | all_risk_only | -4.741317% | -6.577087% to -2.968060% | -5.259127% |
| producer2_controller0 | easy_risk_only | -0.008383% | -0.016183% to -0.002222% | -0.010465% |
| producer2_controller0 | ridge_dual | -0.632447% | -1.227158% to -0.143381% | -0.890677% |
| producer2_controller1 | utility_only | -3.736308% | -4.891665% to -2.551583% | -4.189031% |
| producer2_controller1 | all_risk_only | -3.461553% | -4.598373% to -2.301114% | -3.958106% |
| producer2_controller1 | easy_risk_only | -0.045362% | -0.066910% to -0.024099% | -0.037546% |
| producer2_controller1 | ridge_dual | -0.209602% | -0.600386% to +0.182961% | -0.387525% |

Both risk heads describe exactly the same R-to-P action. Their predicted constraints
do not certify realized risk. No calibration or confirmation localities opened.
No candidate deployment, Stage5C or SMC. Image-pixel annotation-step results only.
