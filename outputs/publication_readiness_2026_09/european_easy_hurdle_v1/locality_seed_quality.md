# Locality, Seed and Tail Readout

Twelve already-opened development localities. The 3 forecast seeds and repeated source-role views are not independent samples. Intervals use 3,000 nominal paired-locality bootstrap draws, not simultaneous or independent-confirmation intervals. No post-readout selection is performed.

| Policy | FDE gain/floor % | Worst-view easy gain/CV % | p95 error ratio/floor | Unknown interventions | Entirely abstaining views |
|---|---:|---:|---:|---:|---:|
| floor | 0 [0, 0] | 0.121432 | 1 [1, 1] | 0 [0, 0] | 216 |
| common_anchor | 0.00521155 [0.00147672, 0.0106108] | 0.176646 | 1.00001 [1, 1.00003] | 3.67593 [0.532407, 9.25463] | 24 |
| raw_independent | 0.385967 [0.218439, 0.579751] | 0.176646 | 0.998429 [0.997324, 0.999395] | 24.8657 [4.81019, 53.4125] | 16 |
| raw_joint | 0.725093 [0.38159, 1.12238] | 0.176646 | 0.995775 [0.993009, 0.998113] | 28.8843 [5.32373, 61.2506] | 16 |
| raw_matched | 0.0347651 [0.018983, 0.0516614] | 0.176646 | 0.999792 [0.999619, 0.999948] | 4.5 [0.574074, 11.7734] | 24 |
| marginal_independent | 0.588653 [0.34757, 0.85575] | -0.0144279 | 0.997447 [0.995293, 0.999571] | 33.3843 [7.73576, 68.361] | 0 |
| marginal_joint | 1.18312 [0.640585, 1.79694] | -0.192985 | 0.993086 [0.987882, 0.997428] | 38.7546 [8.55081, 79.8365] | 0 |
| marginal_matched | 0.0388578 [0.0220936, 0.0560315] | 0.176646 | 0.999775 [0.999605, 0.999929] | 4.45833 [0.611111, 11.5789] | 24 |
| supervised_independent | -0.0035228 [-0.0243426, 0.00852832] | -1.28974 | 1.00033 [0.99999, 1.00094] | 3.99074 [0.810185, 9.63021] | 6 |
| supervised_joint | -0.000283928 [-0.0225353, 0.0137887] | -1.28974 | 1.00028 [0.999909, 1.00092] | 4.03704 [0.810185, 9.7691] | 6 |
| supervised_matched | 0.00860119 [0.00364079, 0.0144558] | 0.176646 | 0.99997 [0.999904, 1.00002] | 3.72685 [0.536921, 9.40278] | 24 |

## Prespecified Primary Contrast

Positive ADE gain means lower error; positive harm reduction means less positive harm. Fixed-floor-denominator harm is a diagnostic, not a replacement for selected-risk. Intervention counts are matched within each query, not only on average.

| Locality | Supervised vs marginal matched ADE gain % | Fixed-denominator harm reduction pp | Intervention difference pp |
|---|---:|---:|---:|
| eu-locality-007 | -0.024466933 | 0.010070576 | 0 |
| eu-locality-008 | -0.058060638 | 0.038699392 | 0 |
| eu-locality-020 | 0.0012413114 | 0.0030787225 | 0 |
| eu-locality-048 | -0.0067147367 | 0.00048451487 | 0 |
| eu-locality-067 | -0.011702367 | 0.0010243287 | 0 |
| eu-locality-074 | -0.0081792715 | 0.0045891664 | 0 |
| eu-locality-082 | -0.0023589967 | 0.0010466231 | 0 |
| eu-locality-110 | -0.044885135 | 0.0025765601 | 0 |
| eu-locality-112 | -0.0014061165 | 5.5872788e-05 | 0 |
| eu-locality-119 | -0.01184768 | 0.0033270207 | 0 |
| eu-locality-124 | -0.022067126 | 0.0018755315 | 0 |
| eu-locality-126 | -0.055157564 | 0.011020598 | 0 |

## All Seeds Retained

| Seed | Policy | ADE gain/floor % | Hard gain/floor % |
|---|---|---:|---:|
| 17 | raw_joint | 0.712766 [0.312293, 1.20847] | 0.820473 [0.35045, 1.38363] |
| 17 | marginal_joint | 0.971255 [0.529762, 1.48645] | 1.09704 [0.527018, 1.73063] |
| 17 | supervised_joint | -0.0240748 [-0.0889506, 0.0116834] | 0.0311953 [0.0047788, 0.0759359] |
| 17 | raw_matched | 0.0348431 [0.0170406, 0.0553558] | 0.0352947 [0.0118588, 0.0632291] |
| 17 | marginal_matched | 0.0320187 [0.0172855, 0.0482014] | 0.027897 [0.00753994, 0.0504148] |
| 17 | supervised_matched | 0.0079733 [0.00244913, 0.0143142] | 0.0041535 [-0.000260186, 0.00937319] |
| 29 | raw_joint | 0.373563 [0.192823, 0.578537] | 0.377595 [0.168453, 0.61415] |
| 29 | marginal_joint | 0.72781 [0.385851, 1.10415] | 0.806177 [0.392199, 1.27981] |
| 29 | supervised_joint | 0.0164968 [0.00393186, 0.0377048] | 0.0184954 [0.00223376, 0.0441575] |
| 29 | raw_matched | 0.0139614 [0.0066684, 0.0237462] | 0.0107677 [0.00143383, 0.0243289] |
| 29 | marginal_matched | 0.015753 [0.00794512, 0.0256381] | 0.0117842 [0.00265556, 0.0252312] |
| 29 | supervised_matched | 0.00356358 [0.00118037, 0.00641369] | 0.00168189 [4.21991e-05, 0.00419716] |
| 43 | raw_joint | 0.437266 [0.257902, 0.635245] | 0.475875 [0.253248, 0.728417] |
| 43 | marginal_joint | 0.844721 [0.468695, 1.27195] | 0.954982 [0.482821, 1.49429] |
| 43 | supervised_joint | 0.0151946 [0.00415802, 0.0308094] | 0.0232998 [0.00282129, 0.0512702] |
| 43 | raw_matched | 0.0220561 [0.0110902, 0.0349332] | 0.0189775 [0.00574855, 0.0349852] |
| 43 | marginal_matched | 0.0290499 [0.0147882, 0.0449614] | 0.0239664 [0.00653054, 0.0421594] |
| 43 | supervised_matched | 0.00398192 [0.00178098, 0.00641006] | 0.00211638 [0.000302955, 0.00492496] |

## Probability and Conditional Costs

Quality scores are query-balanced within known-label held-development data. The marginal arm has no direct occurrence label loss; its probability component need not be calibrated. A better Brier score alone is not proof of better joint allocation or a risk certificate.

| Arm | Easy prevalence | Predicted easy probability | Brier | Log loss | Signed MSE | Signed bias | Conditional reference MSE | Conditional harm MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| marginal | 0.297933 [0.238752, 0.363285] | 0.162181 [0.133246, 0.18772] | 0.262075 [0.218448, 0.317403] | 1.09921 [0.824549, 1.53141] | 0.00418997 [0.00245329, 0.00628605] | 0.00165742 [-0.00417851, 0.0080172] | 14.4046 [8.91464, 23.3563] | 0.0509262 [0.020972, 0.0932788] |
| supervised | 0.297933 [0.238752, 0.363285] | 0.28341 [0.252809, 0.314675] | 0.158131 [0.144101, 0.173253] | 0.499397 [0.454324, 0.546331] | 0.00444302 [0.00267599, 0.00644676] | 0.00891619 [0.00187682, 0.0158769] | 0.00521158 [0.00437707, 0.00606402] | 0.0101517 [0.00600045, 0.0147322] |

No metric, seconds-level, physical-safety, calibration-guarantee or deployment-upgrade claim is inferred. The original incomplete primary remains incomplete; independent roles stay closed.
