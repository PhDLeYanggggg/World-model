# Frozen-Policy Donor Conditioning: Descriptive Readout

All72 views retained; repeated occurrences, not independent samples.
Signed-risk MSE is source-normalized and equal-view averaged. Masses and event diagnostics sum occurrences.

| Arm | All risk MSE | Easy risk MSE | Predicted easy harm | Observed easy harm | Unsupported |
|---|---:|---:|---:|---:|---:|
| origin | 0.0794587451 | 0.000309373416 | 1.37560653 | 95.5606418 | 0 |
| estimation | 0.0780658179 | 0.000309426746 | 2.26819481 | 95.5606418 | 0 |
| pooled_mean | 0.174918744 | 0.00557663629 | 1817.04141 | 95.5606418 | 0 |
| pooled_nearest32 | 0.297406867 | 0.00474068875 | 697.167173 | 95.5606418 | 0 |
| pooled_shuffled32 | 0.216899278 | 0.0113339922 | 1836.23348 | 95.5606418 | 0 |
| selected_mean | 0.0919971944 | 0.00170902231 | 946.449805 | 95.5606418 | 0 |
| selected_nearest32 | 0.101728186 | 0.00236571932 | 370.30528 | 95.5606418 | 0 |
| selected_shuffled32 | 0.101788361 | 0.00238433315 | 959.928872 | 95.5606418 | 0 |

## Event and Magnitude Diagnostics

Unknown labels excluded only from observed scoring. Severity MSE conditions on positive query outcomes
and positive donor-event support; it is a diagnostic, not an inference input or comparable unconditionally.

| Bank / target | Known | Events | Mean event probability | Observed prevalence | Brier | Non-event harm SSE fraction | Positive queries without positive donor |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled/all | 61335 | 12146 | 0.250146735 | 0.198027228 | 0.167976233 | 0.480441879 | 75 |
| pooled/easy | 61335 | 3762 | 0.0609287112 | 0.0613352898 | 0.0581946546 | 0.678322558 | 431 |
| selected/all | 61335 | 12146 | 0.197415831 | 0.198027228 | 0.158534712 | 0.229464107 | 75 |
| selected/easy | 61335 | 3762 | 0.0627460871 | 0.0613352898 | 0.0571716642 | 0.472596511 | 346 |

## Fixed Slices

| Slice | Views | Original easy MSE | Full-bank nearest | Selected nearest | Unsupported |
|---|---:|---:|---:|---:|---:|
| context:single0_seed43_controller1_dimensionless | 12 | 0.000175060151 | 0.00331098308 | 0.0090694894 | 0 |
| context:single0_seed43_controller2_dimensionless | 12 | 0.000128487725 | 0.000997411685 | 0.000404538131 | 0 |
| context:single1_seed43_controller0_dimensionless | 12 | 0.000112149852 | 0.0101727927 | 0.00213228662 | 0 |
| context:single1_seed43_controller2_dimensionless | 12 | 0.000196039928 | 0.00215907572 | 0.000945398727 | 0 |
| context:single2_seed43_controller0_dimensionless | 12 | 0.000598946844 | 0.0072128169 | 0.000635969205 | 0 |
| context:single2_seed43_controller1_dimensionless | 12 | 0.000645555997 | 0.00459105239 | 0.00100663383 | 0 |
| seed:17 | 24 | 0.000313074292 | 0.00511541352 | 0.0023304728 | 0 |
| seed:29 | 24 | 0.000337231259 | 0.00428135944 | 0.00223250914 | 0 |
| seed:43 | 24 | 0.000277814698 | 0.0048252933 | 0.00253417601 | 0 |
| source:eu-locality-007 | 6 | 0.0014372615 | 0.0031685319 | 0.00157729038 | 0 |
| source:eu-locality-008 | 6 | 0.000389529934 | 0.00104495543 | 0.000886543833 | 0 |
| source:eu-locality-020 | 6 | 9.28695137e-06 | 0.022824479 | 0.0172246763 | 0 |
| source:eu-locality-048 | 6 | 0.00016394108 | 0.00133536877 | 0.00343485923 | 0 |
| source:eu-locality-067 | 6 | 1.16620396e-05 | 0.00176274842 | 0.000857472711 | 0 |
| source:eu-locality-074 | 6 | 0.00166629258 | 0.00387546039 | 0.00228220202 | 0 |
| source:eu-locality-082 | 6 | 3.98978762e-08 | 0.0017447151 | 4.05915049e-05 | 0 |
| source:eu-locality-110 | 6 | 7.63244503e-06 | 0.0052802505 | 6.04918088e-05 | 0 |
| source:eu-locality-112 | 6 | 2.46679873e-07 | 0.0012962355 | 0.000688206687 | 0 |
| source:eu-locality-119 | 6 | 9.75146133e-06 | 0.00584118939 | 0.000251150594 | 0 |
| source:eu-locality-124 | 6 | 2.23731743e-07 | 0.00213048537 | 0.000192081886 | 0 |
| source:eu-locality-126 | 6 | 1.66126888e-05 | 0.00658384525 | 0.000893064911 | 0 |

No selection, threshold fit, held/independent inference, policy execution or confirmatory CI.
Obs8/pred12 rawstride12, detector-silver pixel/local only;2% positive selected harm/reference unchanged.
No metric/seconds/gold/physical-safety/true3D/foundation claim. Stage5C/SMC disabled.
