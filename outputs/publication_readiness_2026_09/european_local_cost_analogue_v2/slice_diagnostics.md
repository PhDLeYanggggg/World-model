# Local Quadratic Cost: Fixed-Cohort Readout

72 exposed-TRAIN views; repeated occurrences, not independent samples.
Risk MSE is source-normalized and averaged equally across views, not ADE/FDE.
Masses/SSE sum known normalized occurrences. Unknown queries remain in coverage.

| Arm | All risk MSE | Easy risk MSE | Predicted easy harm | Observed easy harm | B+H>E | EH>H |
|---|---:|---:|---:|---:|---:|---:|
| origin | 0.0794587451 | 0.000309373416 | 1.37560653 | 95.5606418 | 0 | 0 |
| selected_nearest32 | 0.101728186 | 0.00236571932 | 370.30528 | 95.5606418 | 0 | 0 |
| weighted_harm32 | 0.090005651 | 0.000446276429 | 55.366321 | 95.5606418 | 0 | 0 |
| local_signed32 | 0.0896372541 | 0.000442399512 | 55.7726369 | 95.5606418 | 0 | 9019 |
| global_signed_mean | 0.0780496974 | 0.000309445709 | 17.7525983 | 95.5606418 | 0 | 0 |
| shuffled_signed32 | 2.10282241 | 0.00557330058 | 594.829128 | 95.5606418 | 856 | 2015 |

## Fixed Source, Context and Seed Slices

| Slice | Views | Original easy MSE | Uniform nearest | Local signed |
|---|---:|---:|---:|---:|
| source:eu-locality-007 | 6 | 0.0014372615 | 0.00157729038 | 0.00143906113 |
| source:eu-locality-008 | 6 | 0.000389529934 | 0.000886543833 | 0.000403142928 |
| source:eu-locality-020 | 6 | 9.28695137e-06 | 0.0172246763 | 0.00125858091 |
| source:eu-locality-048 | 6 | 0.00016394108 | 0.00343485923 | 0.000180654301 |
| source:eu-locality-067 | 6 | 1.16620396e-05 | 0.000857472711 | 2.57903591e-05 |
| source:eu-locality-074 | 6 | 0.00166629258 | 0.00228220202 | 0.00168956904 |
| source:eu-locality-082 | 6 | 3.98978762e-08 | 4.05915049e-05 | 8.14090484e-05 |
| source:eu-locality-110 | 6 | 7.63244503e-06 | 6.04918088e-05 | 8.66379163e-06 |
| source:eu-locality-112 | 6 | 2.46679873e-07 | 0.000688206687 | 0.000113394767 |
| source:eu-locality-119 | 6 | 9.75146133e-06 | 0.000251150594 | 9.85852953e-06 |
| source:eu-locality-124 | 6 | 2.23731743e-07 | 0.000192081886 | 1.50976037e-06 |
| source:eu-locality-126 | 6 | 1.66126888e-05 | 0.000893064911 | 9.71595726e-05 |
| context:single0_seed43_controller1_dimensionless | 12 | 0.000175060151 | 0.0090694894 | 0.000818034187 |
| context:single0_seed43_controller2_dimensionless | 12 | 0.000128487725 | 0.000404538131 | 0.00012864698 |
| context:single1_seed43_controller0_dimensionless | 12 | 0.000112149852 | 0.00213228662 | 0.000116050063 |
| context:single1_seed43_controller2_dimensionless | 12 | 0.000196039928 | 0.000945398727 | 0.000207874486 |
| context:single2_seed43_controller0_dimensionless | 12 | 0.000598946844 | 0.000635969205 | 0.000600899626 |
| context:single2_seed43_controller1_dimensionless | 12 | 0.000645555997 | 0.00100663383 | 0.00078289173 |
| seed:17 | 24 | 0.000313074292 | 0.0023304728 | 0.000352121175 |
| seed:29 | 24 | 0.000337231259 | 0.00223250914 | 0.000417043302 |
| seed:43 | 24 | 0.000277814698 | 0.00253417601 | 0.000558034058 |

## Leverage and Donor Fit

Minimum effective donors: 1.00013026.
Median of view medians (not overall query median): 7.78266745.
Largest single donor weight: 0.999934876.
Lower donor loss is guaranteed by the constrained local fit, not evidence of prediction lift.
Non-event harm mass is not a classification false-positive rate. Moment violations are not projected away.
All negative slices retained; no posthoc primary substitution or threshold tuning.
No independent CI, held readout or policy execution. obs8/pred12 rawstride12; pixel/local detector-silver.
No metric/seconds/gold/physical-safety/true3D/foundation claim. Stage5C/SMC remain off.
