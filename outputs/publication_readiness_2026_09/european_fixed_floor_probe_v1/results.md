# Fixed-Producer Incremental Probe Results

Fresh_run:108 joint ridge fits,216 five-output linear heads, held scoring and3,000 locality-bootstrap draws.
Cached_verified: nine repaired forecasters and all upstream damping-floor producers.
No new neural forecasters, independent confirmation or deployment. Twelve already-opened development localities.

## Prespecified Primary Contrast

Floor-target safe probe versus matched CV-target safe probe ADE gain (%): **-0.0169 [-0.0218, -0.0115]**.

This tests the target reference with the SAME frozen floor on probe fitting and readout sources.
Four sources fit the forecaster, four fit the floor, two fit each probe and two evaluate it.

## Every Prespecified Action

| Action | ADE gain over floor % | ADE gain over CV % | Easy gain over CV % | Hard gain over floor % | FDE gain over floor % | Intervention fraction |
|---|---:|---:|---:|---:|---:|---:|
| cv | -0.8385 [-1.1435, -0.5427] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -0.4498 [-0.6997, -0.2039] | -1.3752 [-1.8628, -0.9092] | 0.0000 [0.0000, 0.0000] |
| cv_positive | 9.1639 [5.5937, 12.3818] | 9.8957 [6.1946, 13.2808] | -3.7495 [-7.9688, -0.0024] | 12.8448 [9.4974, 15.6455] | 13.2749 [8.5773, 17.3352] | 0.6556 [0.5776, 0.7259] |
| cv_safe | 0.5400 [0.3804, 0.7069] | 1.3599 [0.9498, 1.7383] | 4.4561 [3.4277, 5.4648] | 0.5505 [0.3357, 0.8107] | 0.7058 [0.5263, 0.8943] | 0.0607 [0.0474, 0.0735] |
| floor | 0.0000 [0.0000, 0.0000] | 0.8259 [0.5380, 1.1232] | 3.8347 [2.9864, 4.5881] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| floor_positive | 9.1083 [5.5655, 12.3001] | 9.8410 [6.1466, 13.2006] | -3.5629 [-7.6866, 0.0907] | 12.8206 [9.4832, 15.6099] | 13.2049 [8.5519, 17.2533] | 0.6377 [0.5646, 0.7027] |
| floor_safe | 0.5233 [0.3684, 0.6858] | 1.3434 [0.9384, 1.7175] | 4.3950 [3.3739, 5.3912] | 0.5379 [0.3296, 0.7940] | 0.6817 [0.5051, 0.8646] | 0.0563 [0.0438, 0.0688] |
| oracle_diagnostic | 21.2459 [19.0815, 23.2194] | 21.8909 [19.5424, 23.9774] | 18.8264 [16.3607, 21.1446] | 21.6348 [18.7941, 23.9467] | 25.0575 [21.9360, 27.7124] | 0.6380 [0.6027, 0.6682] |
| parent_calibrated_rebased | 0.1227 [0.0861, 0.1633] | 0.9471 [0.6478, 1.2474] | 6.1613 [4.7633, 7.5209] | 0.0426 [0.0154, 0.0745] | 0.1772 [0.1256, 0.2331] | 0.0962 [0.0754, 0.1171] |
| parent_original | -0.6034 [-0.9023, -0.3255] | 0.2327 [0.1726, 0.3096] | 4.2662 [2.7678, 5.9065] | -0.3858 [-0.6182, -0.1640] | -1.0316 [-1.5261, -0.5682] | 0.1372 [0.1082, 0.1697] |
| parent_rebased | 0.1674 [0.1242, 0.2188] | 0.9916 [0.6800, 1.3025] | 6.5177 [4.8958, 8.1641] | 0.0578 [0.0257, 0.0922] | 0.2307 [0.1682, 0.3106] | 0.1372 [0.1082, 0.1697] |
| raw_neural | 7.7307 [1.8075, 12.3881] | 8.4616 [2.3989, 13.2387] | -11.1692 [-20.8496, -2.7823] | 12.9902 [8.8396, 16.2603] | 12.2825 [4.4115, 18.0533] | 1.0000 [1.0000, 1.0000] |

Floor is the comparator, so its intervention fraction here is0. Probe actions switch from that floor
to neural. Oracle uses future ADE only for diagnosis and carries the chosen trajectory into FDE.
Intervention includes unknown-label rows; their outcomes remain undefined rather than zero.

## Safety and Label Sensitivity

| Action | Worst held-view easy gain over CV % | Zero-CV harmed views | Incremental risk-violating views | Complete-label gain over floor % | Partial-label gain over floor % |
|---|---:|---:|---:|---:|---:|
| cv | 0.0000 | 0 | 0 | -1.1202 [-1.4949, -0.7642] | -0.5112 [-0.7084, -0.3169] |
| cv_positive | -38.3123 | 0 | 216 | 12.1028 [8.3322, 15.8555] | 6.1741 [2.8779, 9.0755] |
| cv_safe | 0.2475 | 0 | 169 | 0.6241 [0.4631, 0.7889] | 0.4372 [0.2798, 0.6138] |
| floor | 0.1214 | 0 | 0 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| floor_positive | -38.3137 | 0 | 216 | 12.0351 [8.2698, 15.7780] | 6.1376 [2.8502, 9.0368] |
| floor_safe | 0.2475 | 0 | 171 | 0.5989 [0.4430, 0.7582] | 0.4297 [0.2748, 0.6075] |
| oracle_diagnostic | 9.9700 | 0 | 0 | 23.4177 [20.8764, 25.9833] | 18.9017 [17.1250, 20.4219] |
| parent_calibrated_rebased | 0.4150 | 0 | 9 | 0.1602 [0.1123, 0.2119] | 0.0837 [0.0531, 0.1216] |
| parent_original | -2.1993 | 0 | 81 | -0.8007 [-1.1869, -0.4438] | -0.3694 [-0.5596, -0.1940] |
| parent_rebased | 0.4150 | 0 | 81 | 0.2239 [0.1650, 0.2988] | 0.1056 [0.0763, 0.1388] |
| raw_neural | -67.7318 | 18 | 216 | 11.3686 [6.2276, 15.9136] | 4.2755 [-1.7825, 8.8333] |

Risk is selected positive incremental harm divided by selected floor error, NOT net degradation.
A null selected ratio means no defined selected reference mass, not certified safety.
There are216 dependent held-locality views per action. Overlapping windows are not independent samples.

## Exact Default-Action Decomposition

All quantities below are percentage points of each locality floor error, then equally averaged.
Original net gain = captured benefit - selected harm - CV fallback regression + CV fallback relief.
Rebased net gain = captured benefit - selected harm. No new decision is fitted for these controls.

| Action | Captured benefit | Selected harm | Missed benefit | Oracle benefit | CV fallback regression | CV fallback relief |
|---|---:|---:|---:|---:|---:|---:|
| floor_positive | 16.9655 [14.2610, 19.5243] | 7.8572 [6.4959, 9.2186] | 4.2804 [3.2128, 5.3873] | 21.2459 [19.0815, 23.2194] | 0.2279 [0.1631, 0.2957] | 0.0157 [0.0110, 0.0206] |
| floor_safe | 0.7054 [0.4963, 0.9429] | 0.1821 [0.1041, 0.2733] | 20.5405 [18.4939, 22.4541] | 21.2459 [19.0815, 23.2194] | 0.8064 [0.5185, 1.1161] | 0.0574 [0.0345, 0.0816] |
| parent_calibrated_rebased | 0.1403 [0.1025, 0.1828] | 0.0177 [0.0130, 0.0228] | 21.1056 [18.9486, 23.1021] | 21.2459 [19.0815, 23.2194] | 0.8548 [0.5379, 1.1802] | 0.0622 [0.0369, 0.0893] |
| parent_rebased | 0.2248 [0.1726, 0.2812] | 0.0574 [0.0349, 0.0817] | 21.0211 [18.8777, 22.9966] | 21.2459 [19.0815, 23.2194] | 0.8320 [0.5203, 1.1533] | 0.0612 [0.0358, 0.0880] |

## Held Prediction Skill

MSE skill is relative to fitting-only source-balanced constant moments with the same causal envelope.
Gain AUROC labels incremental advantage over the frozen floor. These statistics do not certify risk.

| Target reference | Benefit MSE skill % | Harm MSE skill % | Reference MSE skill % | Easy harm MSE skill % | Floor gain AUROC |
|---|---:|---:|---:|---:|---:|
| cv | -7.8650 [-38.3590, 15.2357] | 4.2149 [-3.3815, 10.5841] | 2.6123 [-18.7114, 20.1722] | -2.4144 [-5.6048, -0.2710] | 0.5374 [0.5131, 0.5595] |
| floor | -8.5744 [-40.5037, 15.5515] | 4.2987 [-3.2881, 10.6741] | 2.5346 [-19.2153, 20.2943] | -2.3447 [-5.5445, -0.2324] | 0.5353 [0.5111, 0.5579] |

## Three Forecaster Seeds

| Seed | Parent rebase gain over floor % | Floor positive gain % | Floor safe gain % | CV safe gain % |
|---|---:|---:|---:|---:|
| 17 | 0.1669 [0.1169, 0.2252] | 8.9669 [5.3845, 12.3343] | 0.5339 [0.3868, 0.6928] | 0.5433 [0.3936, 0.7049] |
| 29 | 0.1731 [0.1231, 0.2228] | 9.0562 [5.7032, 12.0797] | 0.5286 [0.3609, 0.7166] | 0.5484 [0.3772, 0.7384] |
| 43 | 0.1623 [0.1196, 0.2220] | 9.3019 [5.6398, 12.5401] | 0.5072 [0.3600, 0.6615] | 0.5281 [0.3767, 0.6827] |

## Gates

- primary_floor_target_beats_matched_cv_target: False
- floor_safe_beats_floor: True
- every_view_easy_preserved: True
- no_zero_reference_harm: True
- no_observed_incremental_risk_violation: False
- independent_confirmation: False
- calibration_certificate: False
- deployment_changed: False
- stage5c_executed: False
- smc_enabled: False

Intervals average dependent seed/producer/fitting-half views within each of12 localities before resampling.
These are unadjusted development intervals, not independent confirmation or simultaneous claims.
All parent damping calibrated-supported actions match this fixed zero-cutoff floor exactly.
A weak linear probe can test this registered repair but cannot prove that no nonlinear causal predictor exists.
Silver image-local obs8/pred12 at raw-frame stride12. Not historical Stage37t50, metric, seconds,
human gold, physical safety, true3D, foundation evidence or submission readiness. Stage5C/SMC remain off.
