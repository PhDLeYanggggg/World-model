# Fixed-Floor Signed-Excess Results

## Material Passport

Fresh108 risk heads and216,000updates; cached_verified108 matched MSE controls, utility/floor and forecasting models.
Twelve development-exposed localities, three forecaster seeds,3,000 locality-bootstrap draws. No new trajectory training or independent test.

## Prespecified Primary

Matched-count MSE minus new selected harm ratio (percentage points): **undefined**.
New ADE gain at equal current-query count (%): **-0.0168 [-0.0539, 0.0169]**.
Counts match within locality/recording/frame including unknown-label rows, never across later queries.

| Policy | ADE / floor gain % | ADE / CV gain % | Hard / floor gain % | Easy / CV gain % | FDE / floor gain % | Intervention % | Selected harm % |
|---|---:|---:|---:|---:|---:|---:|---:|
| excess | 0.1848 [0.1101, 0.2753] | 1.0082 [0.6867, 1.3379] | 0.1360 [0.0518, 0.2428] | 5.2869 [4.1157, 6.3354] | 0.2678 [0.1681, 0.3906] | 6.7818 [5.3441, 8.2383] | undefined |
| floor | 0.0000 [0.0000, 0.0000] | 0.8259 [0.5426, 1.1392] | 0.0000 [0.0000, 0.0000] | 3.8347 [2.9848, 4.5433] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | undefined |
| mse | 0.1544 [0.1160, 0.1968] | 0.9783 [0.6715, 1.3076] | 0.0948 [0.0394, 0.1665] | 5.4587 [4.2867, 6.5396] | 0.2284 [0.1767, 0.2748] | 7.8187 [6.4744, 9.2718] | 3.1126 [2.4950, 3.6932] |
| mse_matched_count | 0.2018 [0.1391, 0.2755] | 1.0249 [0.7032, 1.3647] | 0.1874 [0.0988, 0.2882] | 5.2036 [4.0437, 6.2554] | 0.2963 [0.2083, 0.4033] | 6.7818 [5.3441, 8.2383] | undefined |
| ridge | 0.5233 [0.3833, 0.6913] | 1.3434 [0.9471, 1.7544] | 0.5379 [0.3403, 0.7994] | 4.3950 [3.3327, 5.3373] | 0.6817 [0.5223, 0.8700] | 5.6345 [4.4717, 6.8603] | 4.9104 [3.9138, 6.0153] |
| tail4 | 0.0670 [0.0451, 0.0931] | 0.8920 [0.5975, 1.2168] | 0.0393 [0.0102, 0.0820] | 4.7853 [3.7818, 5.6970] | 0.1006 [0.0689, 0.1332] | 4.9228 [3.9501, 5.9524] | undefined |

Undefined is not zero risk; the fixed roster is not dropped. Count-matched MSE is a ranking diagnostic, not a certified2% policy.

| New versus control | ADE gain % | Harm reduction pp | Intervention difference pp |
|---|---:|---:|---:|
| mse | 0.0306 [-0.0334, 0.1086] | undefined | -1.0369 [-2.0878, 0.0780] |
| mse_matched_count | -0.0168 [-0.0539, 0.0169] | undefined | 0.0000 [0.0000, 0.0000] |
| ridge | -0.3415 [-0.4860, -0.2034] | undefined | 1.1474 [-0.1414, 2.4931] |
| tail4 | 0.1179 [0.0492, 0.2050] | undefined | 1.8590 [0.9307, 2.8363] |

## Worst Views and Label Sensitivity

| Policy | Worst easy gain % | Risk violations /216 | Undefined views | P95 / floor | Complete ADE gain % | Partial ADE gain % | Unknown interventions/view |
|---|---:|---:|---:|---:|---:|---:|---:|
| excess | 0.2431 | 79 | 14 | 0.9990 [0.9983, 0.9997] | 0.2513 [0.1513, 0.3798] | 0.1046 [0.0659, 0.1469] | 20.6389 [4.2910, 44.3010] |
| floor | 0.1214 | 0 | 216 | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| mse | -0.0657 | 110 | 0 | 0.9995 [0.9988, 1.0001] | 0.1962 [0.1478, 0.2466] | 0.1096 [0.0733, 0.1519] | 23.9676 [4.8793, 52.3715] |
| mse_matched_count | 0.2431 | 99 | 14 | 0.9988 [0.9982, 0.9994] | 0.2748 [0.1813, 0.3962] | 0.1159 [0.0814, 0.1575] | 21.7037 [4.2027, 46.6522] |
| ridge | 0.2475 | 171 | 0 | 0.9973 [0.9955, 0.9986] | 0.5989 [0.4565, 0.7626] | 0.4297 [0.2782, 0.6108] | 32.0231 [4.4531, 72.8635] |
| tail4 | 0.3857 | 95 | 3 | 0.9997 [0.9994, 0.9999] | 0.0899 [0.0644, 0.1156] | 0.0440 [0.0226, 0.0714] | 15.5046 [2.8241, 35.3757] |

## Fit Versus Held Signed-Score Errors

| Objective | Fit all MSE | Held all MSE | Fit easy MSE | Held easy MSE | Selected predicted all excess | Selected observed all excess |
|---|---:|---:|---:|---:|---:|---:|
| excess | 0.0730 [0.0560, 0.0934] | 0.1900 [0.1204, 0.2623] | 0.0032 [0.0018, 0.0048] | 0.0027 [0.0018, 0.0038] | undefined | undefined |
| mse | 0.1053 [0.0890, 0.1254] | 0.1836 [0.1164, 0.2528] | 0.0032 [0.0018, 0.0050] | 0.0027 [0.0018, 0.0037] | -0.0102 [-0.0118, -0.0089] | 0.0035 [0.0009, 0.0066] |

Scores/errors use fitting-only mean-CV-cost normalization. Fitting values are repeated as descriptive paired context references, not held measurements or independent fit units.
New output components are unidentified score bases. Do not interpret their ratio as calibrated reference/harm moments. A negative predicted excess is an empirical decision, not an upper confidence bound.

## Three Seeds

| Forecaster seed | New gain / floor % | MSE gain / floor % | New intervention % | New selected harm % |
|---|---:|---:|---:|---:|
| 17 | 0.2831 [0.1335, 0.4848] | 0.1565 [0.1137, 0.2044] | 7.4562 [5.7777, 9.2394] | undefined |
| 29 | 0.1366 [0.0807, 0.1992] | 0.1607 [0.1143, 0.2080] | 6.4767 [5.0044, 7.9914] | undefined |
| 43 | 0.1346 [0.0806, 0.1971] | 0.1460 [0.0984, 0.2073] | 6.4126 [4.9760, 7.8858] | undefined |

## Gates

- primary_equal_count_harm_reduction: False
- equal_count_ADE_advantage: False
- floor_ADE_advantage: True
- nonzero_each_locality: True
- every_view_easy_preserved: True
- no_zero_CV_harm: True
- every_view_defined_risk_within_budget: False
- exploratory_joint_screen_pass: False
- independent_confirmation: False
- calibration_certificate: False
- deployment_changed: False
- stage5c_executed: False
- smc_enabled: False

Per-locality values for every metric are in summary.json. Bootstrap12source means after averaging dependent seed/producer/fit views; overlapping windows are not independent.
All intervals are unadjusted development uncertainty, not confirmation or simultaneous safety. No independent roles, deployment, Stage5C or SMC.
Image-local detector silver; obs8/pred12 rawstride12, not historicalt50, metric, seconds, human gold, physical safety, true3D or foundation.
