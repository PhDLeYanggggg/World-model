# Frozen-Model Source-Gap Results

## Material Passport

Fresh diagnostics of108frozen paired heads; no new training or threshold selection.
Twelve exposed source-training localities, three forecaster seeds. Independent roles remain closed.
Nested held development is not independent confirmation. All predictions and actions match the sealed parent.

## Whole-Role Comparison

| Role / policy | Signed-score MSE | Eligible MSE | Selected MSE | Net floor gain % | Selected harm % | Intervention % |
|---|---:|---:|---:|---:|---:|---:|
| fit/mse | 0.1053 [0.0770, 0.1361] | 0.0856 [0.0646, 0.1058] | 0.0014 [0.0006, 0.0024] | 0.1729 [0.1260, 0.2167] | 1.2795 [1.0307, 1.5422] | 7.2949 [5.8214, 8.7415] |
| fit/excess | 0.0730 [0.0480, 0.1052] | 0.0616 [0.0430, 0.0838] | 0.0008 [0.0004, 0.0014] | 0.2064 [0.1144, 0.3193] | 1.2593 [1.0416, 1.4867] | 7.0401 [4.9124, 9.2212] |
| held/mse | 0.1836 [0.1164, 0.2528] | 0.1544 [0.1041, 0.2001] | 0.0244 [0.0013, 0.0649] | 0.1513 [0.1127, 0.1942] | 3.1073 [2.0309, 4.2484] | 7.8187 [6.4744, 9.2718] |
| held/excess | 0.1900 [0.1204, 0.2623] | 0.1565 [0.1051, 0.2045] | 0.0029 [0.0013, 0.0050] | 0.1749 [0.1104, 0.2507] | 2.1438 [1.5145, 2.8551] | 6.7818 [5.3441, 8.2383] |

Costs use each fitting head's training-only reference-cost scale. Ratios first pool dependent views within locality,
then bootstrap12fixed locality means3,000times. This is NOT the parent primary's mean of view ratios.
It cannot repair the parent's14empty-view failure. In-sample fitting values are not validation evidence.

## All Causal Slices

Low/middle/high boundaries use equal-source fitting-only25th/75th percentiles. Ties may leave empty bins.
Each cell is a development estimate with an unadjusted95%locality interval; not simultaneous inference.

| Axis / bin | New score MSE | New minus MSE score MSE | New harm % | Selected harm share % | Eligible oracle gain % | New floor gain % |
|---|---:|---:|---:|---:|---:|---:|
| mean_step_over_extent/low | 0.0026 [0.0017, 0.0034] | -0.0001 [-0.0002, 0.0001] | 0.7594 [0.5703, 0.9859] | 23.7319 [12.9644, 35.7643] | 12.7911 [10.7308, 14.9161] | 0.7796 [0.5641, 1.0122] |
| mean_step_over_extent/middle | 0.1574 [0.0726, 0.2661] | 0.0011 [-0.0036, 0.0070] | 2.6787 [1.8547, 3.6935] | 49.3046 [41.5271, 57.8859] | 16.7642 [14.4077, 19.2007] | 0.1353 [0.0837, 0.1967] |
| mean_step_over_extent/high | 0.5688 [0.3518, 0.7746] | 0.0188 [-0.0175, 0.0601] | 5.0148 [2.6072, 7.5731] | 26.9635 [14.6585, 40.3889] | 17.3004 [13.9013, 20.2763] | 0.1257 [0.0460, 0.2273] |
| last_step_over_extent/low | 0.0057 [0.0033, 0.0084] | 0.0005 [0.0001, 0.0010] | 1.1011 [0.8625, 1.3322] | 35.0772 [22.1552, 49.1550] | 6.7391 [5.6401, 7.9445] | 0.7235 [0.5312, 0.9226] |
| last_step_over_extent/middle | 0.1551 [0.0779, 0.2474] | 0.0019 [-0.0030, 0.0078] | 2.8469 [1.7438, 4.2880] | 41.5228 [33.0689, 50.3424] | 15.9443 [13.9284, 17.9513] | 0.1463 [0.0872, 0.2157] |
| last_step_over_extent/high | 0.5888 [0.3645, 0.7926] | 0.0153 [-0.0157, 0.0538] | 4.5441 [2.1285, 7.2908] | 23.4000 [11.7725, 36.4716] | 18.2579 [14.4761, 21.8178] | 0.1468 [0.0655, 0.2443] |
| path_nonlinearity/low | 0.4011 [0.2443, 0.5499] | -0.0049 [-0.0208, 0.0108] | 6.8868 [4.0762, 10.3782] | 27.2197 [13.4172, 43.3057] | 16.0575 [13.4699, 18.1547] | 0.0812 [0.0143, 0.1721] |
| path_nonlinearity/middle | 0.2097 [0.1215, 0.3084] | 0.0146 [-0.0018, 0.0385] | 1.8749 [1.4262, 2.4179] | 54.6383 [43.5591, 64.9415] | 14.9604 [11.9713, 17.7825] | 0.1770 [0.1196, 0.2410] |
| path_nonlinearity/high | 0.0300 [0.0135, 0.0503] | -0.0001 [-0.0028, 0.0030] | 0.9306 [0.7099, 1.1380] | 18.1419 [12.0598, 23.9888] | 19.8991 [15.2997, 24.4134] | 0.5104 [0.3470, 0.6878] |
| mean_turn_radians/low | 0.4311 [0.2602, 0.5942] | -0.0075 [-0.0223, 0.0085] | 7.5134 [4.6536, 10.5933] | 28.0624 [13.7672, 44.3224] | 15.8491 [13.1858, 18.0077] | 0.0836 [0.0193, 0.1686] |
| mean_turn_radians/middle | 0.1956 [0.1119, 0.2933] | 0.0155 [0.0009, 0.0389] | 1.8764 [1.3929, 2.4129] | 51.7026 [41.0555, 61.4756] | 15.4294 [12.5557, 18.1523] | 0.1847 [0.1227, 0.2560] |
| mean_turn_radians/high | 0.0153 [0.0093, 0.0219] | -0.0011 [-0.0031, 0.0007] | 0.9857 [0.7702, 1.1958] | 20.2350 [13.7291, 26.2560] | 18.8605 [14.2883, 23.5098] | 0.4866 [0.3437, 0.6391] |
| neighbor_occupancy/low | 0.1338 [0.0755, 0.2062] | 0.0053 [-0.0011, 0.0130] | 1.9195 [1.2418, 2.8183] | 22.2796 [5.1411, 43.6162] | 15.5601 [12.8421, 17.9558] | 0.0827 [0.0457, 0.1283] |
| neighbor_occupancy/middle | 0.1782 [0.1147, 0.2452] | -0.0007 [-0.0083, 0.0068] | 1.6283 [1.0447, 2.3037] | 36.9529 [24.2487, 49.7845] | 16.4182 [13.2647, 19.3570] | 0.1707 [0.0991, 0.2587] |
| neighbor_occupancy/high | undefined | undefined | undefined | 40.7676 [24.9480, 56.3633] | undefined | undefined |
| rollout_disagreement_over_extent/low | 0.0010 [0.0007, 0.0013] | 0.0000 [-0.0000, 0.0001] | 0.9562 [0.7406, 1.1964] | 33.4467 [19.8517, 48.5416] | 6.4931 [5.4743, 7.5842] | 0.6987 [0.5210, 0.8705] |
| rollout_disagreement_over_extent/middle | 0.0836 [0.0407, 0.1362] | -0.0000 [-0.0025, 0.0030] | 3.2378 [2.1114, 4.6198] | 41.6587 [33.7141, 49.6142] | 13.5752 [11.6129, 15.4316] | 0.1168 [0.0692, 0.1734] |
| rollout_disagreement_over_extent/high | 0.7359 [0.4508, 1.0078] | 0.0238 [-0.0106, 0.0638] | 5.2664 [2.6756, 8.0658] | 24.8945 [14.6480, 36.2317] | 20.7825 [16.5044, 24.6656] | 0.1725 [0.0797, 0.2932] |
| feature_radius_over_limit/low | 0.0470 [0.0280, 0.0679] | -0.0017 [-0.0051, 0.0009] | undefined | 11.2946 [6.1502, 17.6829] | 16.5259 [14.3579, 18.4570] | 0.1672 [0.1035, 0.2398] |
| feature_radius_over_limit/middle | 0.1185 [0.0735, 0.1695] | -0.0001 [-0.0055, 0.0049] | 1.9299 [1.3866, 2.5803] | 42.6601 [35.0322, 50.3528] | 17.6247 [15.3217, 19.8008] | 0.1672 [0.1071, 0.2436] |
| feature_radius_over_limit/high | 0.3967 [0.2510, 0.5456] | 0.0228 [-0.0008, 0.0538] | 2.8977 [1.8804, 4.0860] | 46.0453 [35.3415, 56.9667] | 16.4898 [13.0849, 19.9002] | 0.1915 [0.1162, 0.2759] |
| clipped_feature_fraction/low | undefined | undefined | undefined | 0.0000 [0.0000, 0.0000] | undefined | undefined |
| clipped_feature_fraction/middle | undefined | undefined | undefined | 0.0000 [0.0000, 0.0000] | undefined | undefined |
| clipped_feature_fraction/high | 0.1900 [0.1204, 0.2623] | 0.0064 [-0.0023, 0.0180] | 2.1438 [1.5145, 2.8551] | 100.0000 [100.0000, 100.0000] | 16.2717 [13.2406, 19.0725] | 0.1749 [0.1104, 0.2507] |
| existing_support/inside | 0.1836 [0.1167, 0.2537] | 0.0056 [-0.0021, 0.0153] | 2.1438 [1.5145, 2.8551] | 100.0000 [100.0000, 100.0000] | 16.8362 [13.9935, 19.5423] | 0.1795 [0.1157, 0.2546] |
| existing_support/outside | 1.0615 [0.5177, 1.7088] | -0.1495 [-0.6076, 0.1294] | undefined | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

## Evaluation-Only Strata

Future completeness and realized reference errors below never enter inputs or deployment decisions.

| Evaluation stratum | Known row-views | Selected known row-views | Unknown selected row-views | Harm % | Harm share % | Score MSE | New minus MSE score MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference_error_eval_only/unknown | 0 | 0 | 4458 | undefined | 0.0000 [0.0000, 0.0000] | undefined | undefined |
| reference_error_eval_only/low | 1132224 | 213615 | 0 | 2.6016 [2.0110, 3.2571] | 23.8281 [14.3250, 34.8114] | 0.0063 [0.0044, 0.0086] | 0.0003 [-0.0002, 0.0008] |
| reference_error_eval_only/middle | 3055652 | 185495 | 0 | 2.6307 [1.7416, 3.6312] | 49.6338 [41.0624, 58.6502] | 0.0821 [0.0547, 0.1142] | 0.0026 [-0.0004, 0.0060] |
| reference_error_eval_only/high | 1426720 | 23542 | 0 | undefined | 26.5381 [17.0802, 35.5069] | 0.4801 [0.3320, 0.6348] | 0.0152 [-0.0088, 0.0453] |
| label_completeness_eval_only/unknown | 0 | 0 | 4458 | undefined | 0.0000 [0.0000, 0.0000] | undefined | undefined |
| label_completeness_eval_only/partial | 2127906 | 115286 | 0 | 2.0060 [1.3599, 2.8534] | 38.7159 [31.8189, 47.4233] | 0.2372 [0.1510, 0.3284] | 0.0198 [0.0015, 0.0430] |
| label_completeness_eval_only/complete | 3486690 | 307366 | 0 | 2.2698 [1.6396, 2.9190] | 61.2841 [52.5767, 68.1811] | 0.1553 [0.0957, 0.2168] | -0.0021 [-0.0067, 0.0033] |

Counts are repeated role/seed/producer views, not independent trajectories. Unknown outcomes cannot certify safety.
Undefined cells retain the full roster; per-locality numbers and denominators are in summary.json.

## Tied Fitting Boundaries

Equal25th/75th cutpoints by axis (out of108groups): {"mean_step_over_extent": 0, "last_step_over_extent": 0, "path_nonlinearity": 0, "mean_turn_radians": 0, "neighbor_occupancy": 0, "rollout_disagreement_over_extent": 0, "feature_radius_over_limit": 0, "clipped_feature_fraction": 108}.
In particular, a high clipping-fraction bin can include zero clipping when both fitted cuts are zero.
That degeneracy is not evidence that clipping damaged every row. Empty bins are retained, not reassigned.

## Per-Locality Held Results

| Locality | MSE score error | New score error | New minus MSE | New selected harm % | New floor gain % |
|---|---:|---:|---:|---:|---:|
| eu-locality-007 | 0.015476 | 0.015671 | 0.000195 | 4.756604 | 0.346996 |
| eu-locality-008 | 0.210234 | 0.228069 | 0.017836 | 2.634358 | 0.243443 |
| eu-locality-020 | 0.033436 | 0.030676 | -0.002759 | 2.018774 | 0.046063 |
| eu-locality-048 | 0.408631 | 0.392918 | -0.015712 | 0.822759 | 0.121999 |
| eu-locality-067 | 0.263853 | 0.252141 | -0.011713 | 1.934216 | 0.099600 |
| eu-locality-074 | 0.150166 | 0.157096 | 0.006929 | 4.031946 | 0.243128 |
| eu-locality-082 | 0.068988 | 0.071003 | 0.002015 | 1.330444 | 0.055806 |
| eu-locality-110 | 0.164628 | 0.175488 | 0.010860 | 1.052754 | 0.181358 |
| eu-locality-112 | 0.268094 | 0.267518 | -0.000577 | 1.495024 | 0.045675 |
| eu-locality-119 | 0.110390 | 0.118293 | 0.007904 | 2.577737 | 0.457138 |
| eu-locality-124 | 0.343804 | 0.406052 | 0.062248 | 0.861131 | 0.071936 |
| eu-locality-126 | 0.165898 | 0.165039 | -0.000858 | 2.209405 | 0.185088 |

## Limits

This localizes associations, not causal mechanisms or irreducible uncertainty. No bin is promoted to a policy.
Complete detector labels are still silver. Partial future labels can change error interpretation but are unavailable at inference.
Inside a radial support gate does not mean conditional exchangeability or distributional overlap.
Image-local obs8/pred12 rawstride12 only; no metric/seconds/physical-safety/true3D/foundation claim.
No deployment change, independent confirmation, Stage5C execution or SMC.
