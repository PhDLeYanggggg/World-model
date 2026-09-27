# Causal-Descriptor Risk-Head Results

## Material Passport

Fresh training: 108 risk heads, 216,000 updates. Cached_verified: 108 signed-excess controls and the original forecasting/floor/utility chain.
Twelve opened development localities, three forecaster seeds. Independent roles stay closed.
Six causal features and 384 zero-initialized branch parameters added; no loss, budget, draw or deployment change.

## Prespecified Primary

Count-matched control minus new selected harm (pp): **undefined**.
Equal-current-query-count ADE advantage (%): **-0.0139 [-0.0298, -0.0022]**.
Undefined fixed-roster risk is not zero risk. Count matching is diagnostic, not a certified 2% policy.

| Policy | ADE / floor gain % | Hard / floor gain % | Easy / CV gain % | FDE / floor gain % | Intervention % | Selected harm % |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.1848 [0.1101, 0.2753] | 0.1360 [0.0518, 0.2428] | 5.2869 [4.1157, 6.3354] | 0.2678 [0.1681, 0.3906] | 6.7818 [5.3441, 8.2383] | undefined |
| control_matched_count | 0.2731 [0.1706, 0.3918] | 0.2216 [0.0998, 0.3686] | 5.6187 [4.3986, 6.7469] | 0.3930 [0.2547, 0.5562] | 8.0235 [6.4457, 9.5834] | undefined |
| descriptor | 0.2594 [0.1632, 0.3687] | 0.1923 [0.0801, 0.3321] | 5.6406 [4.4125, 6.7641] | 0.3719 [0.2449, 0.5184] | 8.0235 [6.4457, 9.5834] | undefined |
| floor | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 3.8347 [2.9848, 4.5433] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | undefined |
| mse | 0.1544 [0.1160, 0.1968] | 0.0948 [0.0394, 0.1665] | 5.4587 [4.2867, 6.5396] | 0.2284 [0.1767, 0.2748] | 7.8187 [6.4744, 9.2718] | 3.1126 [2.4950, 3.6932] |
| ridge | 0.5233 [0.3833, 0.6913] | 0.5379 [0.3403, 0.7994] | 4.3950 [3.3327, 5.3373] | 0.6817 [0.5223, 0.8700] | 5.6345 [4.4717, 6.8603] | 4.9104 [3.9138, 6.0153] |

| New versus control | ADE gain % | Harm reduction pp | Intervention difference pp |
|---|---:|---:|---:|
| control | 0.0749 [0.0309, 0.1382] | undefined | 1.2417 [0.8509, 1.7612] |
| control_matched_count | -0.0139 [-0.0298, -0.0022] | undefined | 0.0000 [0.0000, 0.0000] |
| mse | 0.1054 [0.0142, 0.2128] | undefined | 0.2048 [-0.9750, 1.4163] |
| ridge | -0.2661 [-0.4028, -0.1424] | undefined | 2.3891 [1.1155, 3.7136] |

## Worst Views and Label Sensitivity

| Policy | Worst easy gain % | Risk violations /216 | Undefined views | Complete ADE gain % | Partial ADE gain % | Unknown interventions /view |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.2431 | 79 | 14 | 0.2513 [0.1513, 0.3798] | 0.1046 [0.0659, 0.1469] | 20.6389 [4.2910, 44.3010] |
| control_matched_count | 0.3716 | 87 | 10 | 0.3567 [0.2267, 0.5074] | 0.1709 [0.1080, 0.2459] | 24.0972 [5.2728, 51.3525] |
| descriptor | 0.3716 | 82 | 10 | 0.3442 [0.2190, 0.4885] | 0.1556 [0.0978, 0.2217] | 23.4954 [5.1706, 49.8024] |
| floor | 0.1214 | 0 | 216 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| mse | -0.0657 | 110 | 0 | 0.1962 [0.1478, 0.2466] | 0.1096 [0.0733, 0.1519] | 23.9676 [4.8793, 52.3715] |
| ridge | 0.2475 | 171 | 0 | 0.5989 [0.4565, 0.7626] | 0.4297 [0.2782, 0.6108] | 32.0231 [4.4531, 72.8635] |

## Training Fit Versus Held Error

| Arm | Fitting all MSE | Held all MSE | Fitting easy MSE | Held easy MSE |
|---|---:|---:|---:|---:|
| control | 0.0730 [0.0560, 0.0934] | 0.1900 [0.1204, 0.2623] | 0.0032 [0.0018, 0.0048] | 0.0027 [0.0018, 0.0038] |
| descriptor | 0.0710 [0.0541, 0.0913] | 0.1874 [0.1193, 0.2583] | 0.0032 [0.0018, 0.0048] | 0.0028 [0.0019, 0.0038] |

Scores divide by fitting-only cost scale. In-sample fit is not validation; score components are not identified cost moments.

## Three Forecaster Seeds

| Seed | New ADE/floor gain % | Control ADE/floor gain % | New harm % |
|---|---:|---:|---:|
| 17 | 0.3692 [0.1788, 0.5986] | 0.2831 [0.1335, 0.4848] | undefined |
| 29 | 0.1907 [0.1244, 0.2639] | 0.1366 [0.0807, 0.1992] | undefined |
| 43 | 0.2183 [0.1378, 0.3098] | 0.1346 [0.0806, 0.1971] | undefined |

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

Bootstrap: 3,000 draws over 12 locality means after dependent producer/fit/seed averaging. Overlapping windows are not independent.
Per-source values are in summary.json. No source is dropped to rescue undefined risk. Development intervals are not safety certificates.
This tests six extra descriptors and 384 parameters together, not their individual semantic value under equal capacity.
Image-local detector silver; obs8/pred12 rawstride12. No metric, seconds, physical-safety, true3D or foundation claim.
No independent confirmation, deployment, Stage5C or SMC.
